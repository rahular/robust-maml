import os
import json
import argparse
import torch
import logging
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.data.data_collator import DefaultDataCollator

from typing import Dict, NamedTuple, Optional
from seqeval.metrics import f1_score, precision_score, recall_score
from data_utils import PosDataset, Split, get_data_config, get_labels
from simple_tagger import PosTagger, get_model_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalPrediction(NamedTuple):
    predictions: np.ndarray
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


def get_optimizers(model, num_warmup_steps, num_training_steps, lr=5e-5):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": model_config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def _training_step(model, inputs, optimizer):
    model.train()
    for k, v in inputs.items():
        inputs[k] = v.to(DEVICE)

    # logger.error(inputs)
    outputs = model(**inputs)
    loss = outputs[0]
    loss.backward()
    return loss.item()


def compute_metrics(p):
    predictions, label_ids = p.predictions, p.label_ids
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def _prediction_loop(model, dataloader, description):
    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", batch_size)
    eval_losses = []
    preds = None
    label_ids = None
    model = model.eval()

    for inputs in tqdm(dataloader, desc=description):
        has_labels = inputs.get("labels") is not None
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                step_eval_loss, logits = outputs[:2]
                eval_losses += [step_eval_loss.mean().item()]
            else:
                logits = outputs[0]

        if preds is None:
            preds = logits.detach()
        else:
            preds = torch.cat((preds, logits.detach()), dim=0)
        if inputs.get("labels") is not None:
            if label_ids is None:
                label_ids = inputs["labels"].detach()
            else:
                label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

    # Finally, turn the aggregated tensors into numpy arrays.
    if preds is not None:
        preds = preds.cpu().numpy()
    if label_ids is not None:
        label_ids = label_ids.cpu().numpy()

    if preds is not None and label_ids is not None:
        metrics = compute_metrics(
            EvalPrediction(predictions=preds, label_ids=label_ids)
        )
    else:
        metrics = {}
    if len(eval_losses) > 0:
        metrics["eval_loss"] = np.mean(eval_losses)

    # Prefix all keys with eval_
    for key in list(metrics.keys()):
        if not key.startswith("eval_"):
            metrics[f"eval_{key}"] = metrics.pop(key)

    return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


def evaluate(loader):
    output = _prediction_loop(model, loader, description="Evaluation")
    return output.metrics


def save(model, optimizer, scheduler):
    save_dir = os.path.join(model_config["output_dir"], args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.th"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"best_optim.th"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, f"best_scheduler.th"))


def init_args():
    parser = argparse.ArgumentParser(
        description="Train POS tagging on various UD datasets"
    )
    parser.add_argument("dataset", help="Dataset to train on")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    data_config = get_data_config()
    model_config = get_model_config()
    dataset_path = data_config[f"{args.dataset}_path"]

    # for reproducibility
    torch.manual_seed(model_config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(model_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])

    # load datasets
    train_dataset = PosDataset(
        dataset_path,
        tokenizer,
        model_config["model_type"],
        model_config["max_seq_length"],
        mode=Split.train,
    )
    dev_dataset = PosDataset(
        dataset_path,
        tokenizer,
        model_config["model_type"],
        model_config["max_seq_length"],
        mode=Split.dev,
    )
    test_dataset = PosDataset(
        dataset_path,
        tokenizer,
        model_config["model_type"],
        model_config["max_seq_length"],
        mode=Split.test,
    )

    # create dataloaders
    batch_size = model_config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        collate_fn=DefaultDataCollator().collate_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=DefaultDataCollator().collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=DefaultDataCollator().collate_batch,
    )

    labels = get_labels(dataset_path)
    label_map = {i: label for i, label in enumerate(labels)}
    model = PosTagger(
        model_config["model_type"], len(labels), model_config["hidden_dropout_prob"]
    )
    model = model.to(DEVICE)

    # create optimizer and lr_scheduler
    num_epochs = model_config["num_epochs"]
    num_training_steps = len(train_loader) * num_epochs
    optimizer, scheduler = get_optimizers(
        model, model_config["num_warmup_steps"], num_training_steps
    )

    best_f1 = 0.0
    best_model_suffix = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        for inputs in epoch_iterator:
            running_loss += _training_step(model, inputs, optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), model_config["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        logger.info(f"Finished epoch {epoch+1} with avg. training loss: {running_loss}")
        # train_f1 = evaluate(train_loader)["eval_f1"]
        dev_f1 = evaluate(dev_loader)["eval_f1"]
        # logger.info(f"Training f1: {train_f1} and validation f1: {dev_f1}")
        logger.info(f"Validation f1: {dev_f1}")

        if dev_f1 > best_f1:
            logger.info("Found new best model!")
            best_f1 = dev_f1
            save(model, optimizer, scheduler)

    model.load_state_dict(
        torch.load(
            os.path.join(model_config["output_dir"], args.dataset, "best_model.th")
        )
    )
    model = model.to(DEVICE)
    test_f1 = evaluate(test_loader)["eval_f1"]
    logger.info(f"Finished training. Test f1: {test_f1}")

