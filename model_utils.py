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
class SeqClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class QuestionAnsweringModelOutput:
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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
            loss_fct = CrossEntropyLoss(reduction="none")
            batch_size, max_len, _ = logits.shape
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

        loss = loss.view([batch_size, max_len]).mean(dim=1)
        return SeqClassifierOutput(loss=loss, logits=logits)


class ClfHead(nn.Module):
    def __init__(self, hidden_dropout_prob, bert_hidden_size):
        super(ClfHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(bert_hidden_size, 1024),
            nn.Dropout(hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(
        self, outputs, labels=None, attention_mask=None,
    ):
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if labels is not None:
            start_positions, end_positions = labels.split(1, dim=-1)
            start_positions = start_positions.squeeze(-1).to(DEVICE)
            end_positions = end_positions.squeeze(-1).to(DEVICE)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )
