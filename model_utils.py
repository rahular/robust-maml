import json
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple

from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            config_json = json.load(f)
        for k, v in config_json.items():
            setattr(self, k, v)


@dataclass
class ClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_type)
        self.bert = BertModel.from_pretrained(self.config.model_type)

    def get_hidden_size(self):
        return self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE), token_type_ids=token_type_ids.to(DEVICE)
        )
        return outputs


class SeqClfHead(nn.Module):
    def __init__(self, num_labels, hidden_dropout_prob, bert_hidden_size):
        super(SeqClfHead, self).__init__()
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(bert_hidden_size, 1024),
            nn.Dropout(hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(1024, num_labels),
        )

    def forward(
        self, outputs, labels=None, attention_mask=None,
    ):
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(DEVICE)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss.to(DEVICE), labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ClassifierOutput(loss=loss, logits=logits,)
        # return {"loss": loss, "logits": logits}

