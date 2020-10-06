import torch
import pyconll
import pandas as pd

from dataclasses import dataclass
from typing import List, Optional
from torch.utils import data
from transformers import AutoTokenizer


@dataclass
class POSInputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class InnerPOSDataset(data.Dataset):
    def __init__(self, data):
        self.input_ids = torch.transpose(torch.stack(data['input_ids']), 0, 1)
        self.attention_mask = torch.transpose(torch.stack(data['attention_mask']), 0, 1)
        self.token_type_ids = torch.transpose(torch.stack(data['token_type_ids']), 0, 1)
        self.label_ids = torch.transpose(torch.stack(data['label_ids']), 0, 1)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.label_ids[idx],
        )


class POS(data.Dataset):
    def __init__(self, path, max_seq_len, model_type):
        sents, labels = [], []
        # path should always be of the form <lang>.<split>
        self.lang = path.split("/")[-1].split(".")[0]
        tagged_sentences = pyconll.load_from_file(path)
        label_set = set()
        for ts in tagged_sentences:
            t, l = [], []
            for token in ts:
                if token.upos and token.form:
                    t.append(token.form)
                    l.append(token.upos)
            label_set.update(l)
            for idx in range(0, len(ts), max_seq_len):
                sents.append(t[idx : idx + max_seq_len])
                labels.append(l[idx : idx + max_seq_len])

        self.label_map = {l: idx for idx, l in enumerate(get_pos_labels())}
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.features = self.convert_examples_to_features(sents, labels, max_seq_len, tokenizer)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            {
                "input_ids": self.features[idx].input_ids,
                "attention_mask": self.features[idx].attention_mask,
                "token_type_ids": self.features[idx].token_type_ids,
                "label_ids": self.features[idx].label_ids,
            },
            self.lang,
        )

    def convert_examples_to_features(
        self,
        sents,
        labels,
        max_seq_len,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputFeatures`
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        features = []
        for _, (sent, lbl) in enumerate(zip(sents, labels)):
            tokens = []
            label_ids = []
            for word, label in zip(sent, lbl):
                word_tokens = tokenizer.tokenize(word)

                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    label_ids.extend([self.label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[: (max_seq_len - special_tokens_count)]
                label_ids = label_ids[: (max_seq_len - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_len - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len
            assert len(label_ids) == max_seq_len

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                POSInputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
                )
            )

        return features


def get_pos_labels():
    return [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]

