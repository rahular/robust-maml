import json
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import BertModel


def get_model_config():
    with open("./model_config.json", "r") as f:
        return json.load(f)


class PosTagger(nn.Module):
    def __init__(self, model_type, num_labels, hidden_dropout_prob):
        super(PosTagger, self).__init__()
        self.model_type = model_type
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_type)
        self.bert.eval()
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.Dropout(hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(1024, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
