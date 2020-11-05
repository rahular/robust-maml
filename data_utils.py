import torch
import pyconll
import logging
import random
import torch.nn as nn
import pandas as pd
import learn2learn as l2l

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.utils import data
from transformers import AutoTokenizer
from learn2learn.data import MetaDataset, TaskDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class POSInputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class InnerPOSDataset(data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.label_ids = data["label_ids"]

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.label_ids[idx],
        )


class CustomPOSLangTaskDataset:
    def __init__(self, datasets, train_type=None):
        self.datasets = {d.lang: d for d in datasets}
        self.id2lang = {idx: lang for idx, lang in enumerate(sorted(self.datasets.keys()))}
        self.lang2id = {lang: idx for idx, lang in self.id2lang.items()}
        total_langs = len(self.datasets)
        self.tau = nn.Parameter(
            torch.ones(total_langs, dtype=torch.float32, requires_grad=(train_type != "metabase"), device=DEVICE)
            / total_langs
        )
        self.activation = lambda x: x  # pass-through activation
        if train_type == "minmax":
            self.activation = F.softmax
        elif train_type == "constrain":
            self.activation = F.softplus

    def _append_tensor(self, X, x):
        X["input_ids"].append(torch.tensor(x["input_ids"], dtype=torch.long))
        X["attention_mask"].append(torch.tensor(x["attention_mask"], dtype=torch.long))
        X["token_type_ids"].append(torch.tensor(x["token_type_ids"], dtype=torch.long))
        X["label_ids"].append(torch.tensor(x["label_ids"], dtype=torch.long))
        return X

    def _stack_tensors(self, X):
        X["input_ids"] = torch.stack(X["input_ids"], 0)
        X["attention_mask"] = torch.stack(X["attention_mask"], 0)
        X["token_type_ids"] = torch.stack(X["token_type_ids"], 0)
        X["label_ids"] = torch.stack(X["label_ids"], 0)
        return X

    def sample(self, k=50, langs=None):
        tau_dist = Categorical(logits=self.tau)
        if langs is None:
            lang_idx = tau_dist.sample(sample_shape=[k])
            langs = [self.id2lang[idx.item()] for idx in lang_idx]
        else:
            # In some languages, there are no development sets
            lang_idx = []
            for idx, lang in enumerate(langs):
                if lang not in self.lang2id:
                    lang = random.choice(list(self.lang2id.keys()))
                    langs[idx] = lang
                lang_idx.append(self.lang2id[lang])
            lang_idx = torch.tensor(lang_idx)

        X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": []}
        Y = []
        for lang in langs:
            x, y = self.datasets[lang].sample()
            X = self._append_tensor(X, x)
            Y.append(y)
        X = self._stack_tensors(X)

        return (X, Y), self.activation(self.tau[lang_idx]) + (0 * self.tau.sum())

    def test_sample(self, k=50):
        # this is used only during testing, so there should only be one language
        assert len(self.datasets) == 1
        dataset = list(self.datasets.values())[0]
        support_X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": []}
        query_X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": []}
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        support_ids, query_ids = ids[:k], ids[k:]

        for idx in support_ids:
            x, _ = dataset[idx]
            support_X = self._append_tensor(support_X, x)
        support_X = self._stack_tensors(support_X)

        for idx in query_ids:
            x, _ = dataset[idx]
            query_X = self._append_tensor(query_X, x)
        query_X = self._stack_tensors(query_X)

        return support_X, query_X


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

    def sample(self):
        return self.__getitem__(random.randint(0, len(self.features) - 1))

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

