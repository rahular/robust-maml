import torch
import numpy as np
import torch.nn as nn

import data_utils

from datetime import datetime
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score

savedir = None


def get_savedir_name():
    global savedir
    if not savedir:
        savedir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return savedir


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


def compute_loss(loader, bert_model, learner, label_map):
    loss = 0.0
    for batch, features in enumerate(loader):
        # it is done this way to be consistent with both outer (regular) and inner (meta) dataloaders
        input_ids, attention_mask, token_type_ids, labels = features[0], features[1], features[2], features[3]
        with torch.no_grad():
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)
        output = learner(bert_output, labels=labels, attention_mask=attention_mask)
        curr_loss = output.loss
        loss += curr_loss

        gold, preds = None, None
        for lgt, lbl in zip(output.logits, labels):
            if preds is None:
                preds = torch.unsqueeze(lgt.detach(), 0)
            else:
                preds = torch.cat((preds, torch.unsqueeze(lgt.detach(), 0)), dim=0)
            if gold is None:
                gold = torch.unsqueeze(lbl.detach(), 0)
            else:
                gold = torch.cat((gold, torch.unsqueeze(lbl.detach(), 0)), dim=0)

    metrics = None
    if preds is not None and gold is not None:
        metrics = compute_metrics(preds.cpu().numpy(), gold.cpu().numpy(), label_map)

    loss /= batch + 1
    return loss, metrics
