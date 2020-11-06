import os
import json
import argparse
import torch
import logging

import utils
import data_utils
import model_utils

import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_shot_evaluate(test_set, label_map, bert_model, clf_head, config):
    loader = DataLoader(test_set, batch_size=config.batch_size, collate_fn=utils.pos_collate_fn)
    loss, metrics = utils.compute_loss_metrics(loader, bert_model, clf_head, label_map, grad_required=False)
    metrics.update({"loss": loss.mean().item()})
    return metrics


def init_args():
    parser = argparse.ArgumentParser(description="Test POS tagging on in-domain UD datasets")
    parser.add_argument("--model_path", dest="model_path", type=str, help="Path of the model to load", required=True)
    return parser.parse_args()


def main():
    args = init_args()
    config_path = os.path.join(args.model_path, "config.json")
    load_path = os.path.join(args.model_path, "best_model.th")
    logger.info("Loading config from path: {}".format(config_path))
    logger.info("Loading model from path: {}".format(load_path))

    config = model_utils.Config(config_path)
    torch.manual_seed(config.seed)

    data_dir = "./data/pos/indomain"
    test_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".test")]
    
    if "/pos/" in data_dir:
        data_class = data_utils.POS
        label_map = {idx: l for idx, l in enumerate(data_utils.get_pos_labels())}
    elif "/ner/" in data_dir:
        data_class = data_utils.NER
        label_map = {idx: l for idx, l in enumerate(data_utils.get_ner_labels())}
    else:
        raise ValueError(f"Unknown task or incorrect `config.data_dir`: {config.data_dir}")

    bert_model = model_utils.BERT(config)
    bert_model = bert_model.eval().to(DEVICE)
    clf_head = model_utils.SeqClfHead(len(label_map), config.hidden_dropout_prob, bert_model.get_hidden_size())
    clf_head.load_state_dict(torch.load(load_path))
    clf_head = clf_head.to(DEVICE)

    summary_metrics = pd.DataFrame(columns=["lang", "p", "r", "f"])
    for idx, test_path in enumerate(sorted(test_paths)):
        dataset_name = test_path.split("/")[-1].split(".")[0]
        logger.info(f"Testing {dataset_name}...")
        test_set = data_class(test_path, config.max_seq_length, config.model_type)
        r = zero_shot_evaluate(test_set, label_map, bert_model, clf_head, config)
        summary_metrics.loc[idx] = [dataset_name, r["precision"], r["recall"], r["f1"]]

    save_dir = os.path.join(args.model_path, "result")
    os.makedirs(save_dir, exist_ok=True)
    summary_metrics.to_csv(os.path.join(save_dir, "indomain.csv"))


if __name__ == "__main__":
    main()
