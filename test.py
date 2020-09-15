import os
import json
import argparse
import torch
import logging
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
from meta_data_utils import PosDataset, Split, get_data_config, read_examples_from_file
from simple_tagger import BERT, Classifier, get_model_config
from simple_trainer import compute_metrics, EvalPrediction

import learn2learn as l2l

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
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
    for (
        iters,
        (input_ids, attention_mask, token_type_ids, labels, class_index),
    ) in enumerate(
        torch.utils.data.DataLoader(
            _BatchedDataset(task), batch_size=batch_size, shuffle=True, num_workers=0
        )
    ):

        input_ids = torch.stack([str_to_tensor(x) for x in input_ids]).to(DEVICE)
        attention_mask = torch.stack([str_to_tensor(x) for x in attention_mask]).to(
            DEVICE
        )
        token_type_ids = torch.stack([str_to_tensor(x) for x in token_type_ids]).to(
            DEVICE
        )
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
                preds[ci] = torch.cat(
                    (preds[ci], torch.unsqueeze(lgt.detach(), 0)), dim=0
                )
            if ci not in label_ids:
                label_ids[ci] = torch.unsqueeze(lbl.detach(), 0)
            else:
                label_ids[ci] = torch.cat(
                    (label_ids[ci], torch.unsqueeze(lbl.detach(), 0)), dim=0
                )
    # from here there are no more batches
    for class_index in preds.keys():
        preds[class_index] = preds[class_index].cpu().numpy()
    for class_index in label_ids.keys():
        label_ids[class_index] = label_ids[class_index].cpu().numpy()

    metrics = {}
    for class_index in label_ids.keys():
        if preds[class_index] is not None and label_ids[class_index] is not None:
            metrics[class_index] = compute_metrics(
                EvalPrediction(
                    predictions=preds[class_index], label_ids=label_ids[class_index]
                ),
                label_map,
            )
    return loss / (iters + 1), metrics


def evaluate(
    data_gen, meta_model, bert_model, task_bs, inner_loop_steps, inner_lr, num_episodes
):
    task_query_error = 0.0
    tqdm_bar = tqdm(range(num_episodes))
    all_metrics = {}
    for episode in tqdm_bar:  # episode loop = task loop
        # clone the meta model for use in inner loop. Back-propagating losses on the cloned module will populate the
        # buffers of the original module
        learner = meta_model.clone()
        # sample train and validation tasks
        support_task, query_task = data_gen.sample(), data_gen.sample()

        # Inner Loop
        for step in range(inner_loop_steps):  # inner loop
            support_error, _ = compute_loss(
                support_task, bert_model, learner, batch_size=task_bs
            )
            grads = torch.autograd.grad(
                support_error,
                learner.parameters(),
                create_graph=True,
                allow_unused=True,
            )
            l2l.algorithms.maml_update(learner, inner_lr, grads)

        # Compute validation loss / query loss
        query_error, metrics = compute_loss(
            query_task, bert_model, learner, batch_size=task_bs
        )
        task_query_error += query_error
        tqdm_bar.set_description("Query Loss: {:.3f}".format(query_error.item()))
        for class_index in metrics.keys():
            if class_index in all_metrics:
                all_metrics[class_index]["p"] += metrics[class_index]["precision"]
                all_metrics[class_index]["r"] += metrics[class_index]["recall"]
                all_metrics[class_index]["f"] += metrics[class_index]["f1"]
            else:
                all_metrics[class_index] = {
                    "p": metrics[class_index]["precision"],
                    "r": metrics[class_index]["recall"],
                    "f": metrics[class_index]["f1"],
                }

    for class_index in all_metrics.keys():
        all_metrics[class_index]["p"] /= num_episodes
        all_metrics[class_index]["r"] /= num_episodes
        all_metrics[class_index]["f"] /= num_episodes
    all_metrics["loss"] = (task_query_error / num_episodes).item()
    return all_metrics


def get_model_load_path():
    return os.path.join(
        model_config["output_dir"],
        "posbert_i_{}_o_{}_s_{}_d_{}".format(
            model_config["inner_lr"],
            model_config["outer_lr"],
            model_config["shots"],
            args.datasets,
        ),
    )


def init_args():
    parser = argparse.ArgumentParser(
        description="Test POS tagging on various UD datasets"
    )
    parser.add_argument("--datasets", help="Dataset(s)")
    parser.add_argument(
        "--dataset_type", help="Type of data to evaluate on: train, test, dev"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    data_config = get_data_config()
    model_config = get_model_config()

    datasets = args.datasets.split(",")

    dataset_paths = []
    for dataset in datasets:
        dataset_path = data_config[f"{dataset}_path"]
        dataset_paths.append(dataset_path)

    # for reproducibility
    torch.manual_seed(model_config["seed"])
    np.random.seed(model_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])

    if args.dataset_type == "test":
        data_split = Split.test
    if args.dataset_type == "train":
        data_split = Split.train
    if args.dataset_type == "dev":
        data_split = Split.dev

    labels = set()
    datasets = []
    class_index = 0

    # build label set for all datasets
    for dataset_path in dataset_paths:
        _, l = read_examples_from_file(
            dataset_path, Split.train, model_config["max_seq_length"]
        )
        labels.update(l)
    labels = sorted(list(labels))
    label_map = {i: label for i, label in enumerate(labels)}

    # load individual datasets
    for dataset_path in dataset_paths:
        dataset = PosDataset(
            class_index,
            dataset_path,
            labels,
            tokenizer,
            model_config["model_type"],
            model_config["max_seq_length"],
            mode=data_split,
        )
        datasets.append(dataset)
        class_index += 1

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
        num_tasks=20000,
        task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(meta_dataset, n=ways, k=shots),
            l2l.data.transforms.LoadData(meta_dataset),
        ],
    )

    # define the bert and postagger model
    bert_model = BERT(model_config["model_type"])
    bert_model.eval()
    bert_model = bert_model.to(DEVICE)

    postagger = Classifier(
        len(labels), model_config["hidden_dropout_prob"], bert_model.get_hidden_size(),
    )
    if model_config["is_load"]:
        postagger.load_state_dict(
            torch.load(os.path.join(get_model_load_path(), "best_model.th"))
        )
    postagger.to(DEVICE)

    num_episodes = model_config["num_episodes"]
    task_bs = model_config["task_batch_size"]
    inner_loop_steps = model_config["inner_loop_steps"]
    inner_lr = model_config["inner_lr"]

    if model_config["is_fomaml"]:
        meta_model = l2l.algorithms.MAML(postagger, lr=inner_lr, first_order=True)
    else:
        meta_model = l2l.algorithms.MAML(postagger, lr=inner_lr)

    all_metrics = evaluate(
        data_gen,
        meta_model,
        bert_model,
        task_bs,
        inner_loop_steps,
        inner_lr,
        num_episodes,
    )

    logger.info(json.dumps(all_metrics, indent=2))

