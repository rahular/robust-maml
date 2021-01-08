"""
The code in this file is heavily borrowed from https://github.com/huggingface/transformers
"""
import os
import json
import torch
import transformers
import pyconll
import logging
import random
import torch.nn as nn
import pandas as pd
import learn2learn as l2l

from tqdm import tqdm
from functools import partial
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional
from multiprocessing import Pool, cpu_count
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.utils import data
from transformers import AutoTokenizer
from transformers.data.processors.squad import squad_convert_example_to_features, SquadExample
from learn2learn.data import MetaDataset, TaskDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class InputFeatures:
    unique_id: int
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class InnerDataset(data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.label_ids = data["label_ids"]
        self.unique_ids = data["unique_ids"]

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.label_ids[idx],
            self.unique_ids[idx]
        )


class CustomLangTaskDataset(nn.Module):
    def __init__(self, datasets, train_type=None):
        super(CustomLangTaskDataset, self).__init__()
        if isinstance(datasets[0], data.Subset):
            self.datasets = {d.dataset.lang: d for d in datasets}
        else:
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
        X["unique_ids"].append(x["unique_id"])
        return X

    def _stack_tensors(self, X):
        X["input_ids"] = torch.stack(X["input_ids"], 0)
        X["attention_mask"] = torch.stack(X["attention_mask"], 0)
        X["token_type_ids"] = torch.stack(X["token_type_ids"], 0)
        X["label_ids"] = torch.stack(X["label_ids"], 0)
        X["unique_ids"] = torch.tensor(X["unique_ids"], dtype=torch.int)
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

        X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": [], "unique_ids": []}
        Y = []
        for lang in langs:
            if isinstance(self.datasets[lang], data.Subset):
                x, y = self.datasets[lang].dataset.sample()
            else:
                x, y = self.datasets[lang].sample()
            X = self._append_tensor(X, x)
            Y.append(y)
        X = self._stack_tensors(X)

        return (X, Y), self.activation(self.tau[lang_idx]) + (0 * self.tau.sum())

    def test_sample(self, k=50):
        # this is used only during testing, so there should only be one language
        assert len(self.datasets) == 1
        dataset = list(self.datasets.values())[0]
        support_X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": [], "unique_ids": []}
        query_X = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label_ids": [], "unique_ids": []}
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


##################### sequence labeling data utils #####################
class POS(data.Dataset):
    def __init__(self, path, max_seq_len, model_type, max_count=10 ** 6):
        sents, labels = [], []
        count = 0
        # filename should always be of the form <lang>.<split>
        self.lang = path.split("/")[-1].split(".")[0]
        cached_features_file = path + f".{max_count}.th"
        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Saving features into cached file {cached_features_file}")
            tagged_sentences = pyconll.load_from_file(path)
            for ts in tagged_sentences:
                t, l = [], []
                for token in ts:
                    if token.upos and token.form:
                        t.append(token.form)
                        l.append(token.upos)
                for idx in range(0, len(ts), max_seq_len):
                    sents.append(t[idx : idx + max_seq_len])
                    labels.append(l[idx : idx + max_seq_len])
                count += 1
                if count > max_count:
                    break
            label_map = {l: idx for idx, l in enumerate(get_pos_labels())}
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            self.features = convert_examples_to_features(sents, labels, label_map, max_seq_len, tokenizer)
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            {
                "input_ids": self.features[idx].input_ids,
                "attention_mask": self.features[idx].attention_mask,
                "token_type_ids": self.features[idx].token_type_ids,
                "label_ids": self.features[idx].label_ids,
                "unique_id": self.features[idx].unique_id,
            },
            self.lang,
        )

    def sample(self):
        return self.__getitem__(random.randint(0, len(self.features) - 1))


class NER(data.Dataset):
    def __init__(self, path, max_seq_len, model_type, max_count=10 ** 6):
        # path should always be of the form <lang>.<split>
        self.lang = path.split("/")[-1].split(".")[0]
        sents, labels = [], []
        words, tags = [], []
        cached_features_file = path + f".{max_count}.th"
        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        assert len(words) == len(tags)
                        for idx in range(0, len(tags), max_seq_len):
                            sents.append(words[idx : idx + max_seq_len])
                            labels.append(tags[idx : idx + max_seq_len])
                        words, tags = [], []
                        # we don't count and break here, because the first k examples are not necessarily
                        # the best as they may come only from a handful of wiki docs.
                    else:
                        parts = line.split()
                        words.append(parts[0])
                        tags.append(parts[-1])
                if len(words) > 0:
                    assert len(words) == len(tags)
                    for idx in range(0, len(tags), max_seq_len):
                        sents.append(words[idx : idx + max_seq_len])
                        labels.append(tags[idx : idx + max_seq_len])
            concat_list = list(zip(sents, labels))
            random.shuffle(concat_list)
            sents, labels = zip(*concat_list)
            sents, labels = list(sents[:max_count]), list(labels[:max_count])
            label_map = {l: idx for idx, l in enumerate(get_ner_labels())}
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            self.features = convert_examples_to_features(sents, labels, label_map, max_seq_len, tokenizer)
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            {
                "input_ids": self.features[idx].input_ids,
                "attention_mask": self.features[idx].attention_mask,
                "token_type_ids": self.features[idx].token_type_ids,
                "label_ids": self.features[idx].label_ids,
                "unique_id": self.features[idx].unique_id,
            },
            self.lang,
        )

    def sample(self):
        return self.__getitem__(random.randint(0, len(self.features) - 1))


def convert_examples_to_features(
    sents,
    labels,
    label_map,
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
    for idx, (sent, lbl) in enumerate(zip(sents, labels)):
        tokens = []
        label_ids = []
        for word, label in zip(sent, lbl):
            word_tokens = tokenizer.tokenize(word)

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

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
            InputFeatures(
                unique_id=idx, input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
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


def get_ner_labels():
    return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]


##################### QA data utils #####################
class QA(data.Dataset):
    def __init__(self, path, max_clen, max_qlen, doc_stride, model_type):
        # filename should always be of the form <lang>.<split>
        self.lang = path.split("/")[-1].split(".")[0]
        cached_features_file = path + f"_{max_clen}_{doc_stride}_{max_qlen}.th"
        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            features_and_dataset = torch.load(cached_features_file)
            self.features, self.dataset, self.examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            logger.info(f"Saving features into cached file {cached_features_file}")
            self.examples = create_squad_examples(path)
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            self.features, self.dataset = squad_convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=max_clen,
                doc_stride=doc_stride,
                max_query_length=max_qlen,
            )
            torch.save(
                {"features": self.features, "dataset": self.dataset, "examples": self.examples}, cached_features_file
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # start and end positions are combined as `label_ids` so that the flow doesn't break
        # we'll split it up later inside the model
        return (
            {
                "input_ids": self.features[idx].input_ids,
                "attention_mask": self.features[idx].attention_mask,
                "token_type_ids": self.features[idx].token_type_ids,
                "label_ids": (self.features[idx].start_position, self.features[idx].end_position),
                "unique_id": self.features[idx].unique_id
            },
            self.lang,
        )

    def sample(self):
        return self.__getitem__(random.randint(0, len(self.features) - 1))


def squad_convert_example_to_features_init(tokenizer_for_convert):
    transformers.data.processors.squad.tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=True,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    dataset = data.TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_feature_index,
        all_start_positions,
        all_end_positions,
        all_cls_index,
        all_p_mask,
        all_is_impossible,
    )
    return features, dataset


def create_squad_examples(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)["data"]
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                    answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
    return examples
