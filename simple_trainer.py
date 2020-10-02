import os
import json
import argparse
import torch
import logging
import wandb
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.data.data_collator import DefaultDataCollator

from typing import Dict, NamedTuple, Optional
from seqeval.metrics import f1_score, precision_score, recall_score
from data_utils import PosDataset, Split, get_data_config, read_examples_from_file
from simple_tagger import BERT, Classifier, get_model_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
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
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": model_config["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def _training_step(bert, model, inputs, optimizer):
    model.train()
    for k, v in inputs.items():
        inputs[k] = v.to(DEVICE)
    with autocast():
        # outputs = model(**inputs)
        with torch.no_grad():
            bert_output = bert(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        outputs = model(bert_output, labels=inputs["labels"], attention_mask=inputs["attention_mask"])
    loss = outputs[0]
    grad_scaler.scale(loss).backward()
    return loss.item()


def compute_metrics(p, label_map):
    predictions, label_ids = p.predictions, p.label_ids
    preds = np.argmax(predictions, axis=-1)
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


def _prediction_loop(bert, model, dataloader, label_map, description):
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
            # outputs = model(**inputs)
            bert_output = bert(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
            outputs = model(bert_output, labels=inputs["labels"], attention_mask=inputs["attention_mask"])
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
        metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids), label_map)
    else:
        metrics = {}
    if len(eval_losses) > 0:
        metrics["eval_loss"] = np.mean(eval_losses)

    # Prefix all keys with eval_
    for key in list(metrics.keys()):
        if not key.startswith("eval_"):
            metrics[f"eval_{key}"] = metrics.pop(key)

    return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


def evaluate(loader, label_map, bert, model):
    output = _prediction_loop(bert, model, loader, label_map, description="Evaluation")
    return output.metrics


def save(model, optimizer, scheduler, last_epoch):
    save_dir = wandb.run.dir
    logger.info("Saving model checkpoint to %s", save_dir)
    # save model config as well
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        f.write(json.dumps(model_config, indent=2))
    # save model weigths
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.th"))
    # save training state if required
    torch.save(
        {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(), "last_epoch": last_epoch},
        os.path.join(save_dir, "optim.th"),
    )


def write_logs(metrics, epoch, name="train"):
    wandb.log(
        {
            "loss/{}_loss".format(name): metrics["eval_loss"],
            "metrics/{}_prf".format(name): {
                "precision": metrics["eval_precision"],
                "recall": metrics["eval_recall"],
                "f1_score": metrics["eval_f1"],
            },
        }
    )


def init_args():
    parser = argparse.ArgumentParser(description="Train POS tagging on various UD datasets")
    parser.add_argument("datasets", metavar="datasets", type=str, nargs="+", help="Datasets to train on")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    data_config = get_data_config()
    model_config = get_model_config()

    wandb.init(project="nlp-meta-learning")
    wandb.config.update(model_config)
    wandb.config.update(vars(args))

    dataset_paths = []
    for dataset in args.datasets:
        dataset_paths.append(data_config[f"{dataset}_path"])

    # for reproducibility
    torch.manual_seed(model_config["seed"])
    np.random.seed(model_config["seed"])

    # for mixed precision training
    grad_scaler = GradScaler()

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"])

    # create a universal label set from the training files of all datasets
    labels = set()
    for dataset_path in dataset_paths:
        _, l = read_examples_from_file(dataset_path, Split.train, model_config["max_seq_length"])
        labels.update(l)
    labels = sorted(list(labels))

    # read data splits
    train_datasets, dev_datasets, test_datasets = [], [], []
    for dataset_path in dataset_paths:
        train_datasets.append(
            PosDataset(
                dataset_path,
                labels,
                tokenizer,
                model_config["model_type"],
                model_config["max_seq_length"],
                mode=Split.train,
            )
        )
        dev_datasets.append(
            PosDataset(
                dataset_path,
                labels,
                tokenizer,
                model_config["model_type"],
                model_config["max_seq_length"],
                mode=Split.dev,
            )
        )
        test_datasets.append(
            PosDataset(
                dataset_path,
                labels,
                tokenizer,
                model_config["model_type"],
                model_config["max_seq_length"],
                mode=Split.test,
            )
        )
    train_datasets = ConcatDataset(train_datasets)
    dev_datasets = ConcatDataset(dev_datasets)

    # create dataloaders
    batch_size = model_config["batch_size"]
    train_loader = DataLoader(
        train_datasets,
        batch_size=batch_size,
        sampler=RandomSampler(train_datasets),
        collate_fn=DefaultDataCollator().collate_batch,
    )
    dev_loader = DataLoader(
        dev_datasets,
        batch_size=batch_size,
        sampler=SequentialSampler(dev_datasets),
        collate_fn=DefaultDataCollator().collate_batch,
    )

    label_map = {i: label for i, label in enumerate(labels)}
    # model = PosTagger(model_config["model_type"], len(labels), model_config["hidden_dropout_prob"])
    bert = BERT(model_config["model_type"])
    bert.eval()
    bert = bert.to(DEVICE)
    model = Classifier(len(labels), model_config["hidden_dropout_prob"], bert.get_hidden_size())

    wandb.watch(model)
    model = model.to(DEVICE)

    # create optimizer and lr_scheduler
    num_epochs = model_config["num_epochs"]
    num_training_steps = len(train_loader) * num_epochs
    optimizer, scheduler = get_optimizers(model, model_config["num_warmup_steps"], num_training_steps)

    best_f1, patience_ctr = 0.0, 0
    best_state_dict = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        for training_step, inputs in enumerate(epoch_iterator):
            step_loss = _training_step(bert, model, inputs, optimizer)
            running_loss += step_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config["max_grad_norm"])
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            model.zero_grad()
        logger.info(f"Finished epoch {epoch+1} with avg. training loss: {running_loss/len(inputs)}")
        wandb.log({"loss/running_loss": running_loss / len(inputs)})

        # train_metrics = evaluate(train_loader, label_map, bert, model)
        # write_logs(train_metrics, epoch, "train")
        dev_metrics = evaluate(dev_loader, label_map, bert, model)
        write_logs(dev_metrics, epoch, "validation")

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

    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, "best_model.th")))
    model = model.to(DEVICE)
    train_metrics = evaluate(train_loader, label_map, bert, model)
    dev_metrics = evaluate(dev_loader, label_map, bert, model)
    test_metrics = {}
    for idx, test_data in enumerate(test_datasets):
        logging.info("Testing on {}...".format(args.datasets[idx]))
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=SequentialSampler(test_data),
            collate_fn=DefaultDataCollator().collate_batch,
        )
        test_metrics[args.datasets[idx]] = evaluate(test_loader, label_map, bert, model)

    # dump results to file and stdout
    final_result = {
        "train": train_metrics,
        "validation": dev_metrics,
        "test": test_metrics,
        # "num_epochs": epoch,
    }
    wandb.run.summary["final_results"] = final_result
    final_result = json.dumps(final_result, indent=2)
    with open(os.path.join(wandb.run.dir, "result.json"), "w") as f:
        f.write(final_result)
    logger.info(f"Final result: {final_result}")

