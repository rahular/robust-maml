import os
import json
import argparse
import torch
import logging

# import wandb
import statistics as stat
import torch.nn as nn
import numpy as np

from torch import optim
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers.data.data_collator import DefaultDataCollator

from typing import Dict, NamedTuple, Optional
from seqeval.metrics import f1_score, precision_score, recall_score
from data_utils import PosDataset as RegularPosDataset
from meta_data_utils import PosDataset as MetaPosDataset
from meta_data_utils import Split, get_data_config, read_examples_from_file
from simple_tagger import BERT, Classifier, get_model_config
from simple_trainer import compute_metrics, EvalPrediction
from simple_trainer import evaluate as regular_evaluate

import learn2learn as l2l

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
# wandb.init(project="nlp-meta-learning")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _BatchedDataset(torch.utils.data.Dataset):
    def __init__(self, batched):
        self.input_ids = [s for s in batched[0][0]]
        self.attention_mask = [s for s in batched[0][1]]
        self.token_type_ids = [s for s in batched[0][2]]
        self.labels = [s for s in batched[0][3]]
        self.class_index = [y for y in batched[1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
            self.class_index[idx],
        )


def str_to_tensor(x):
    return torch.tensor(list(map(int, x.split(" "))))


def compute_loss(task, bert_model, learner, batch_size):
    loss = 0.0
    preds, label_ids = {}, {}
    for (iters, (input_ids, attention_mask, token_type_ids, labels, class_index),) in enumerate(
        torch.utils.data.DataLoader(_BatchedDataset(task), batch_size=batch_size, shuffle=True, num_workers=0)
    ):

        input_ids = torch.stack([str_to_tensor(x) for x in input_ids]).to(DEVICE)
        attention_mask = torch.stack([str_to_tensor(x) for x in attention_mask]).to(DEVICE)
        token_type_ids = torch.stack([str_to_tensor(x) for x in token_type_ids]).to(DEVICE)
        labels = torch.stack([str_to_tensor(x) for x in labels]).to(DEVICE)

        with torch.no_grad():
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
        outputs = learner(bert_output, labels=labels, attention_mask=attention_mask)
        step_eval_loss, logits = outputs[:2]
        loss += step_eval_loss
        # unstack batches into individual items
        for ci, lgt, lbl in zip(class_index, logits, labels):
            ci = ci.item()
            if ci not in preds:
                preds[ci] = torch.unsqueeze(lgt.detach(), 0)
            else:
                preds[ci] = torch.cat((preds[ci], torch.unsqueeze(lgt.detach(), 0)), dim=0)
            if ci not in label_ids:
                label_ids[ci] = torch.unsqueeze(lbl.detach(), 0)
            else:
                label_ids[ci] = torch.cat((label_ids[ci], torch.unsqueeze(lbl.detach(), 0)), dim=0)
    # from here there are no more batches
    for class_index in preds.keys():
        preds[class_index] = preds[class_index].cpu().numpy()
    for class_index in label_ids.keys():
        label_ids[class_index] = label_ids[class_index].cpu().numpy()

    metrics = {}
    for class_index in label_ids.keys():
        if preds[class_index] is not None and label_ids[class_index] is not None:
            metrics[class_index] = compute_metrics(
                EvalPrediction(predictions=preds[class_index], label_ids=label_ids[class_index]), label_map,
            )
    return loss / (iters + 1), metrics


def meta_evaluate(datsets, bert_model, postagger):
    # concatenate individual datasets into a single dataset
    combined_dataset = ConcatDataset(datasets)

    # convert to metadataset which is suitable for sampling tasks in an episode
    meta_dataset = l2l.data.MetaDataset(combined_dataset)

    # shots = number of examples per task, ways = number of classes per task
    shots = model_config["shots"]
    ways = model_config["ways"]

    # create task generators
    data_gen = l2l.data.TaskDataset(
        meta_dataset,
        num_tasks=model_config["num_tasks"],
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(meta_dataset, n=ways, k=shots),
            l2l.data.transforms.LoadData(meta_dataset),
        ],
    )
    num_episodes = model_config["num_episodes"]
    task_bs = model_config["task_batch_size"]
    inner_loop_steps = model_config["inner_loop_steps"]
    inner_lr = model_config["inner_lr"]

    if model_config["is_fomaml"]:
        meta_model = l2l.algorithms.MAML(postagger, lr=inner_lr, first_order=True)
    else:
        meta_model = l2l.algorithms.MAML(postagger, lr=inner_lr)

    task_support_error, task_query_error = 0.0, []
    tqdm_bar = tqdm(range(num_episodes))
    all_metrics = {}
    for _ in tqdm_bar:  # episode loop = task loop
        # clone the meta model for use in inner loop. Back-propagating losses on the cloned module will populate the
        # buffers of the original module
        learner = meta_model.clone()
        # sample train and validation tasks
        support_task, query_task = data_gen.sample(), data_gen.sample()

        # Inner Loop
        for _ in range(inner_loop_steps):  # inner loop
            support_error, _ = compute_loss(support_task, bert_model, learner, batch_size=task_bs)
            grads = torch.autograd.grad(support_error, learner.parameters(), create_graph=True, allow_unused=True)
            l2l.algorithms.maml_update(learner, inner_lr, grads)
            task_support_error += support_error
        # Compute validation loss / query loss
        query_error, metrics = compute_loss(query_task, bert_model, learner, batch_size=task_bs)
        task_query_error.append(query_error)
        tqdm_bar.set_description("Query Loss: {:.3f}".format(query_error.item()))
        # wandb.log({"support_loss": support_error / inner_loop_steps, "query_loss": query_error})
        for class_index in metrics.keys():
            if class_index in all_metrics:
                all_metrics[class_index]["p"].append(metrics[class_index]["precision"])
                all_metrics[class_index]["r"].append(metrics[class_index]["recall"])
                all_metrics[class_index]["f"].append(metrics[class_index]["f1"])
            else:
                all_metrics[class_index] = {
                    "p": [metrics[class_index]["precision"]],
                    "r": [metrics[class_index]["recall"]],
                    "f": [metrics[class_index]["f1"]],
                }
            # wandb.log({"{}_{}".format(dataset_names[class_index], k): v for k, v in metrics[class_index].items()})

    summary_metrics = {}
    for class_index in all_metrics.keys():
        summary_metrics[dataset_names[class_index]] = {
            "p_stdev": stat.stdev(all_metrics[class_index]["p"]),
            "p": stat.mean(all_metrics[class_index]["p"]),
            "r_stdev": stat.stdev(all_metrics[class_index]["r"]),
            "r": stat.mean(all_metrics[class_index]["r"]),
            "f_stdev": stat.stdev(all_metrics[class_index]["f"]),
            "f": stat.mean(all_metrics[class_index]["f"]),
        }
    summary_metrics["loss"] = torch.tensor(task_query_error).mean().item()
    # wandb.run.summary["summary_metrics"] = summary_metrics
    return summary_metrics


def init_args():
    parser = argparse.ArgumentParser(description="Test POS tagging on various UD datasets")
    parser.add_argument("datasets", metavar="datasets", type=str, nargs="+", help="Datasets to test on")
    parser.add_argument("-s", "--split", help="Data split to evaluate on", default="test")
    parser.add_argument(
        "-e", "--eval_type", help="Type of evaluation (meta/regular)", choices=["meta", "regular", "both"], default="both"
    )
    # pylint: disable=unused-argument
    req_named_params = parser.add_argument_group("required named arguments")
    req_named_params.add_argument("-m", "--model_path", help="path of the model to load", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    data_config = get_data_config()
    model_config = get_model_config()

    dataset_names = args.datasets

    dataset_paths = []
    for dataset in dataset_names:
        dataset_path = data_config[f"{dataset}_path"]
        dataset_paths.append(dataset_path)

    # for reproducibility
    torch.manual_seed(model_config["seed"])
    np.random.seed(model_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])

    if args.split == "test":
        data_split = Split.test
    elif args.split == "train":
        data_split = Split.train
    elif args.split == "dev":
        data_split = Split.dev

    labels = set()
    # build label set for all datasets
    for dataset_path in dataset_paths:
        _, l = read_examples_from_file(dataset_path, Split.train, model_config["max_seq_length"])
        labels.update(l)
    labels = sorted(list(labels))
    label_map = {i: label for i, label in enumerate(labels)}

    if args.eval_type == "both":
        args.eval_type = ["regular", "meta"]
    else:
        args.eval_type = [args.eval_type]

    for eval_type in args.eval_type:
        datasets = []
        # load individual datasets
        for class_index, dataset_path in enumerate(dataset_paths):
            dataset_args = [
                class_index,
                dataset_path,
                labels,
                tokenizer,
                model_config["model_type"],
                model_config["max_seq_length"],
                data_split,
            ]
            if eval_type == "regular":
                dataset = RegularPosDataset(*dataset_args[1:])
            else:
                dataset = MetaPosDataset(*dataset_args)
            datasets.append(dataset)

        # define the bert and postagger model
        bert_model = BERT(model_config["model_type"])
        bert_model.eval()
        bert_model = bert_model.to(DEVICE)
        postagger = Classifier(len(labels), model_config["hidden_dropout_prob"], bert_model.get_hidden_size())

        load_path = os.path.join(args.model_path, "best_model.th")
        logging.info("Loading model from path: {}".format(load_path))
        postagger.load_state_dict(torch.load(load_path))
        postagger.to(DEVICE)

        logging.info("Running {} evaluation".format(eval_type))
        if eval_type == "regular":
            summary_metrics = {}
            for idx, data in enumerate(datasets):
                loader = DataLoader(
                    data,
                    batch_size=model_config["batch_size"],
                    sampler=SequentialSampler(data),
                    collate_fn=DefaultDataCollator().collate_batch,
                )
                summary_metrics[args.datasets[idx]] = regular_evaluate(loader, label_map, bert_model, postagger)
        else:
            summary_metrics = meta_evaluate(datasets, bert_model, postagger)
        logger.info(json.dumps(summary_metrics, indent=2))

