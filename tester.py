import os
import json
import argparse
import torch
import logging

import utils
import data_utils
import model_utils

import statistics as stat
import learn2learn as l2l

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_evaluate(test_set, label_map, bert_model, clf_head, config):
    loader = DataLoader(test_set, batch_size=config.batch_size, collate_fn=utils.pos_collate_fn)
    loss, metrics = utils.compute_loss_metrics(loader, bert_model, clf_head, label_map)
    metrics.update({"loss": loss.item()})
    return metrics


def meta_evaluate(dataset, label_map, bert_model, clf_head, config):
    meta_dataset = l2l.data.MetaDataset(dataset)
    test_task = l2l.data.TaskDataset(
        meta_dataset,
        num_tasks=config.num_tasks,
        task_transforms=[
            # here n=1 because the labels are language ids
            l2l.data.transforms.FusedNWaysKShots(meta_dataset, n=1, k=config.shots),
            l2l.data.transforms.LoadData(meta_dataset),
        ],
    )
    num_episodes = config.num_episodes
    task_bs = config.task_batch_size
    inner_loop_steps = config.inner_loop_steps
    inner_lr = config.inner_lr

    meta_model = l2l.algorithms.MAML(clf_head, lr=inner_lr, first_order=config.is_fomaml)

    task_support_error, task_query_error = 0.0, []
    tqdm_bar = tqdm(range(num_episodes))
    all_metrics = {"p": [], "r": [], "f": []}
    for _ in tqdm_bar:
        learner = meta_model.clone()
        support_task, _ = test_task.sample()
        query_task, _ = test_task.sample()

        for _ in range(inner_loop_steps):
            support_loader = DataLoader(
                data_utils.InnerPOSDataset(support_task), batch_size=task_bs, shuffle=True, num_workers=0
            )
            support_error, _ = utils.compute_loss_metrics(support_loader, bert_model, learner, label_map)
            grads = torch.autograd.grad(support_error, learner.parameters(), create_graph=True, allow_unused=True)
            l2l.algorithms.maml_update(learner, inner_lr, grads)
            task_support_error += support_error

        query_loader = DataLoader(
            data_utils.InnerPOSDataset(query_task), batch_size=task_bs, shuffle=True, num_workers=0
        )
        query_error, metrics = utils.compute_loss_metrics(query_loader, bert_model, learner, label_map)
        task_query_error.append(query_error)
        tqdm_bar.set_description("Query Loss: {:.3f}".format(query_error.item()))

        all_metrics["p"].append(metrics["precision"])
        all_metrics["r"].append(metrics["recall"])
        all_metrics["f"].append(metrics["f1"])

    all_metrics["p_stdev"] = stat.stdev(all_metrics["p"])
    all_metrics["p"] = stat.mean(all_metrics["p"])
    all_metrics["r_stdev"] = stat.stdev(all_metrics["r"])
    all_metrics["r"] = stat.mean(all_metrics["r"])
    all_metrics["f_stdev"] = stat.stdev(all_metrics["f"])
    all_metrics["f"] = stat.mean(all_metrics["f"])
    return all_metrics


def init_args():
    parser = argparse.ArgumentParser(description="Test POS tagging on various UD datasets")
    parser.add_argument("--test_path", dest="test_path", type=str, help="Datasets to test on", required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, help="Path of the model to load", required=True)
    parser.add_argument(
        "-e",
        "--eval_type",
        help="Type of evaluation (meta/regular)",
        choices=["meta", "full", "both"],
        default="both",
    )
    return parser.parse_args()


def main():
    args = init_args()
    config = model_utils.Config(os.path.join(args.model_path, "config.json"))

    # for reproducibility
    torch.manual_seed(config.seed)

    test_set = data_utils.POS(args.test_path, config.max_seq_length, config.model_type)
    label_map = {idx: l for idx, l in enumerate(data_utils.get_pos_labels())}

    if args.eval_type == "both":
        eval_types = ["full", "meta"]
    else:
        eval_types = [args.eval_type]
    for eval_type in eval_types:
        bert_model = model_utils.BERT(config)
        bert_model = bert_model.eval().to(DEVICE)
        clf_head = model_utils.SeqClfHead(len(label_map), config.hidden_dropout_prob, bert_model.get_hidden_size())

        load_path = os.path.join(args.model_path, "best_model.th")
        logging.info("Loading model from path: {}".format(load_path))
        clf_head.load_state_dict(torch.load(load_path))
        clf_head = clf_head.to(DEVICE)

        logging.info("Running {} evaluation".format(eval_type))
        if eval_type == "full":
            summary_metrics = full_evaluate(test_set, label_map, bert_model, clf_head, config)
        elif eval_type == "meta":
            summary_metrics = meta_evaluate(test_set, label_map, bert_model, clf_head, config)
        logger.info(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
