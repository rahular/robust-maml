import os
import math
import torch
import string
import random
import logging
import numpy as np
import torch.nn as nn

import data_utils

from datetime import datetime
from collections import OrderedDict
from torch.utils import data
from seqeval.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
savedir = None


def get_savedir_name():
    global savedir
    if not savedir:
        savedir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # just to be extra careful
        savedir += "-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        logging.info(f"All models will be saved at: {savedir}")
    return savedir


def set_savedir_name(name):
    global savedir
    savedir = name


def clean_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def compute_metrics(predictions, label_ids, label_map):
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


def compute_loss_metrics(loader, bert_model, learner, label_map, grad_required=True, return_metrics=True):
    loss = None
    gold, preds = None, None
    for features in loader:
        # it is done this way to be consistent with both outer (regular) and inner (meta) dataloaders
        input_ids, attention_mask, token_type_ids, labels = features[0], features[1], features[2], features[3]
        with torch.set_grad_enabled(bert_model.training and grad_required):
            # We want to set bert_model to train mode even if it is passed in eval model.
            # Otherwise, dropout, etc. will not work. Hence get the mode, so that we can set it later.
            is_bert_training = bert_model.training
            bert_model = bert_model.train()
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
            if not is_bert_training:
                bert_model = bert_model.eval()
        with torch.set_grad_enabled(grad_required):
            output = learner(bert_output, labels=labels, attention_mask=attention_mask)
        if loss is None:
            loss = output.loss
        else:
            loss = torch.cat([loss, output.loss], 0)

        if return_metrics and label_map is not None:  # HACK: easiest way to identify if the task not sequence labeling
            for lgt, lbl in zip(output.logits, labels):
                if preds is None:
                    preds = torch.unsqueeze(lgt.detach().cpu(), 0)
                else:
                    preds = torch.cat((preds, torch.unsqueeze(lgt.detach().cpu(), 0)), dim=0)
                if gold is None:
                    gold = torch.unsqueeze(lbl.detach().cpu(), 0)
                else:
                    gold = torch.cat((gold, torch.unsqueeze(lbl.detach().cpu(), 0)), dim=0)

    metrics = None
    if preds is not None and gold is not None:
        metrics = compute_metrics(preds.numpy(), gold.numpy(), label_map)

    return loss, metrics


def qa_evaluate(lang, examples, features, model_type, loader, bert_model, learner, save_dir):
    all_results, loss, uids = [], [], []
    for batch in loader:
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, labels, unique_ids = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
            )
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
            outputs = learner(bert_output, labels=labels, attention_mask=attention_mask)
            loss.append(outputs.loss.mean().item())

        for i, uid in enumerate(unique_ids):
            unique_id = int(uid.item())
            start_logits = outputs.start_logits[i].detach().cpu().tolist()
            end_logits = outputs.end_logits[i].detach().cpu().tolist()
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)
            uids.append(unique_id)

    save_dir = os.path.join(save_dir, "result")
    os.makedirs(save_dir, exist_ok=True)
    output_prediction_file = os.path.join(save_dir, f"{lang}.predictions")
    output_nbest_file = os.path.join(save_dir, f"{lang}.nbest_predictions")
    features = [f for f in features if f.unique_id in uids]
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=False,
        output_prediction_file=output_prediction_file,
        output_nbest_file=output_nbest_file,
        output_null_log_odds_file=None,
        verbose_logging=True,
        version_2_with_negative=False,
        null_score_diff_threshold=-np.inf,
        tokenizer=AutoTokenizer.from_pretrained(model_type),
    )
    results = squad_evaluate(examples, predictions)
    return torch.tensor(loss), dict(results)


def collate_fn(batch):
    input_ids, attention_mask, token_type_ids, label_ids, unique_ids, languages = [], [], [], [], [], []
    for f, l in batch:
        input_ids.append(f["input_ids"])
        attention_mask.append(f["attention_mask"])
        token_type_ids.append(f["token_type_ids"])
        label_ids.append(f["label_ids"])
        unique_ids.append(f["unique_id"])
        languages.append(l)
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(token_type_ids),
        torch.tensor(label_ids),
        torch.tensor(unique_ids),
        languages,
    )


class BalancedTaskSampler(data.sampler.Sampler):
    """
    Code taken from: https://towardsdatascience.com/unbalanced-data-loading-for-multi-task-learning-in-pytorch-e030ad5033b
    iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.features) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = data.sampler.RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


def clip_grad_norm(grads, max_norm):
    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2).to(device) for grad in grads]), 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = [grad.detach().mul_(clip_coef.to(grad.device)) for grad in grads]
    return tuple(grads)
