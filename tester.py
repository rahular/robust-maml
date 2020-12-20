import os
import json
import argparse
import torch
import logging
import copy

import utils
import data_utils
import model_utils

import statistics as stat
import learn2learn as l2l

from collections import defaultdict
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_evaluate(test_set, label_map, bert_model, clf_head, config, args):
    loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=utils.collate_fn)
    bert_model.eval().to(DEVICE)
    clf_head.eval().to(DEVICE)
    if label_map is not None:
        loss, metrics = utils.compute_loss_metrics(
            loader, bert_model, clf_head, label_map, grad_required=False, return_metrics=True
        )
    else:
        loss, metrics = utils.qa_evaluate(
            args.test_lang,
            test_set.examples,
            test_set.features,
            config.model_type,
            loader,
            bert_model,
            clf_head,
            args.model_path,
        )
    metrics.update({"loss": loss.mean().item()})
    return metrics


def evaluate(test_set, label_map, bert_model, clf_head, config, args, shots):
    task = data_utils.CustomLangTaskDataset([test_set])
    num_episodes = config.num_episodes
    task_bs = config.task_batch_size
    inner_loop_steps = config.inner_loop_steps
    inner_lr = config.inner_lr

    task_support_error = 0.0
    tqdm_bar = tqdm(range(num_episodes))
    all_metrics = defaultdict(list)

    for _ in tqdm_bar:
        learner = copy.deepcopy(clf_head).to(DEVICE)
        encoder = copy.deepcopy(bert_model).to(DEVICE)
        optimizer = optim.SGD(list(learner.parameters()) + list(encoder.parameters()), lr=inner_lr)
        support_task, query_task = task.test_sample(k=shots)
        encoder.train()
        learner.train()
        for _ in range(inner_loop_steps):
            support_loader = DataLoader(
                data_utils.InnerDataset(support_task), batch_size=task_bs, shuffle=True, num_workers=0
            )
            support_error, _ = utils.compute_loss_metrics(
                support_loader, encoder, learner, label_map, grad_required=True, return_metrics=False
            )
            support_error = support_error.mean()
            support_error.backward()
            optimizer.step()
            optimizer.zero_grad()
            task_support_error += support_error.item()

        encoder.eval()
        learner.eval()
        query_loader = DataLoader(
            data_utils.InnerDataset(query_task), batch_size=task_bs, shuffle=False, num_workers=0
        )
        if label_map is not None:
            query_error, metrics = utils.compute_loss_metrics(
                query_loader, encoder, learner, label_map, grad_required=False, return_metrics=True
            )
            all_metrics["p"].append(metrics["precision"])
            all_metrics["r"].append(metrics["recall"])
            all_metrics["f"].append(metrics["f1"])
        else:
            query_error, metrics = utils.qa_evaluate(
                args.test_lang,
                test_set.examples,
                test_set.features,
                config.model_type,
                query_loader,
                encoder,
                learner,
                args.model_path,
            )
            all_metrics["exact"].append(metrics["exact"])
            all_metrics["f1"].append(metrics["f1"])
        tqdm_bar.set_description("Query Loss: {:.3f}".format(query_error.mean().item()))

    if label_map is not None:
        all_metrics["p_stdev"] = stat.stdev(all_metrics["p"])
        all_metrics["p"] = stat.mean(all_metrics["p"])
        all_metrics["r_stdev"] = stat.stdev(all_metrics["r"])
        all_metrics["r"] = stat.mean(all_metrics["r"])
        all_metrics["f_stdev"] = stat.stdev(all_metrics["f"])
        all_metrics["f"] = stat.mean(all_metrics["f"])
    else:
        all_metrics["exact_stdev"] = stat.stdev(all_metrics["exact"])
        all_metrics["exact"] = stat.mean(all_metrics["exact"])
        all_metrics["f1_stdev"] = stat.stdev(all_metrics["f1"])
        all_metrics["f1"] = stat.mean(all_metrics["f1"])
    return all_metrics


def init_args():
    parser = argparse.ArgumentParser(description="Test POS tagging on various UD datasets")
    parser.add_argument("--test_lang", dest="test_lang", type=str, help="Language to test on", required=True)
    parser.add_argument("--model_path", dest="model_path", type=str, help="Path of the model to load", required=True)
    # parser.add_argument(
    #     "--shots", dest="shots", type=int, help="Number of examples to use for finetuning", required=True
    # )
    return parser.parse_args()


def main():
    args = init_args()
    config_path = os.path.join(args.model_path, "config.json")
    load_encoder_path = os.path.join(args.model_path, "best_encoder.th")
    load_head_path = os.path.join(args.model_path, "best_model.th")
    logging.info("Loading config from path: {}".format(config_path))
    if os.path.isfile(load_encoder_path):
        logging.info("Loading encoder from path: {}".format(load_encoder_path))
    logging.info("Loading model from path: {}".format(load_head_path))

    config = model_utils.Config(config_path)
    torch.manual_seed(config.seed)

    data_dir = config.data_dir
    test_path = os.path.join(data_dir, f"{args.test_lang}.test")

    if "/pos/" in data_dir:
        data_class = data_utils.POS
        label_map = {idx: l for idx, l in enumerate(data_utils.get_pos_labels())}
    elif "/ner/" in data_dir:
        data_class = data_utils.NER
        label_map = {idx: l for idx, l in enumerate(data_utils.get_ner_labels())}
    elif "/tydiqa" in data_dir or "squad" in data_dir:
        data_class = data_utils.QA
        label_map = None
    else:
        raise ValueError(f"Unknown task or incorrect `config.data_dir`: {config.data_dir}")

    bert_model = model_utils.BERT(config)
    if label_map is not None:
        test_set = data_class(test_path, config.max_seq_length, config.model_type)
        clf_head = model_utils.SeqClfHead(len(label_map), config.hidden_dropout_prob, bert_model.get_hidden_size())
    else:
        test_set = data_class(test_path, config.max_clen, config.max_qlen, config.doc_stride, config.model_type)
        clf_head = model_utils.ClfHead(config.hidden_dropout_prob, bert_model.get_hidden_size())
    if os.path.isfile(load_encoder_path):
        bert_model.load_state_dict(utils.clean_keys(torch.load(load_encoder_path)))
    clf_head.load_state_dict(utils.clean_keys(torch.load(load_head_path)))

    # shots = args.shots
    shots = [0, 1, 2, 5, 10]
    summary_metrics = {}
    for shot in shots:
        if shot == 0:
            summary_metrics["0"] = zero_shot_evaluate(test_set, label_map, bert_model, clf_head, config, args)
        else:
            summary_metrics[str(shot)] = evaluate(test_set, label_map, bert_model, clf_head, config, args, shot)

    save_dir = os.path.join(args.model_path, "result")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "{}.json".format(args.test_lang)), "w") as f:
        f.write(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
