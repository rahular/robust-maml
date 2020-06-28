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
from torch.utils.tensorboard import SummaryWriter
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


def get_model_save_path():
    return os.path.join(
        model_config["output_dir"],
        "{}_{}".format(model_config["model_type"], args.dataset),
    )


def save(model, optimizer, scheduler, last_epoch):
    save_dir = get_model_save_path()
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", save_dir)
    # save model config as well
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        f.write(json.dumps(model_config, indent=2))
    # save model weigths
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.th"))
    # save training state if required
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "last_epoch": last_epoch,
        },
        os.path.join(save_dir, "optim.th"),
    )


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
    np.random.seed(model_config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])
    labels = get_labels(dataset_path)

    # load datasets
    train_dataset = PosDataset(
        dataset_path,
        labels,
        tokenizer,
        model_config["model_type"],
        model_config["max_seq_length"],
        mode=Split.train,
    )
    dev_dataset = PosDataset(
        dataset_path,
        labels,
        tokenizer,
        model_config["model_type"],
        model_config["max_seq_length"],
        mode=Split.dev,
    )
    test_dataset = PosDataset(
        dataset_path,
        labels,
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

    # for tensorboard logging
    tb_writer = SummaryWriter(os.path.join(get_model_save_path(), "logs"))

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

    best_f1, patience_ctr = 0.0, 0
    best_state_dict = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        for training_step, inputs in enumerate(epoch_iterator):
            step_loss = _training_step(model, inputs, optimizer)
            running_loss += step_loss
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), model_config["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            tb_writer.add_scalar(
                "loss/step_loss", step_loss, epoch * len(epoch_iterator) + training_step
            )
        logger.info(
            f"Finished epoch {epoch+1} with avg. training loss: {running_loss/len(inputs)}"
        )
        dev_metrics = evaluate(dev_loader)

        tb_writer.add_scalar("loss/train_loss", running_loss / len(inputs), epoch)
        tb_writer.add_scalar("loss/dev_loss", dev_metrics["eval_loss"], epoch)
        tb_writer.add_scalar(
            "metrics/dev_precision", dev_metrics["eval_precision"], epoch
        )
        tb_writer.add_scalar("metrics/dev_recall", dev_metrics["eval_recall"], epoch)
        tb_writer.add_scalar("metrics/dev_f1", dev_metrics["eval_f1"], epoch)
        logger.info("Validation f1: {}".format(dev_metrics["eval_f1"]))

        if dev_metrics["eval_f1"] > best_f1:
            logger.info("Found new best model!")
            best_f1 = dev_metrics["eval_f1"]
            save(model, optimizer, scheduler, epoch)
            best_state_dict = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr == model_config["patience"]:
                logger.info("Ran out of patience. Stopping training early...")
                break

    model.load_state_dict(
        torch.load(os.path.join(get_model_save_path(), "best_model.th"))
    )
    model = model.to(DEVICE)
    train_metrics = evaluate(train_loader)
    dev_metrics = evaluate(dev_loader)
    test_metrics = evaluate(test_loader)

    # dump results to file and stdout
    final_result = json.dumps(
        {
            "train": train_metrics,
            "validation": dev_metrics,
            "test": test_metrics,
            "num_epochs": epoch,
        },
        indent=2,
    )
    with open(os.path.join(get_model_save_path(), "result.json"), "w") as f:
        f.write(final_result)
    logger.info(f"Final result: {final_result}")

