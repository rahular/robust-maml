"""
To  download TydiQA, do:
>> mkdir -p data/tydiqa && cd data/tydiqa
>> wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json
>> wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.tgz
>> tar -xf tydiqa-goldp-v1.1-dev.tgz
>> python make_qa_data.py (execute this file)
"""
import os
import json
import random
import numpy as np

import unicodedata as u

from collections import defaultdict
from shutil import copyfile

random.seed(42)
save_dir = "./data/tydiqa/all"
test_path = "./data/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-{}.json"


def count(data):
    qs = 0
    qlens, clens = [], []
    for dp in data:
        for p in dp["paragraphs"]:
            clens.append(len(p["context"]))
            qs += len(p["qas"])
            for qa in p["qas"]:
                qlens.append(len(qa["question"]))
    return qs, np.mean(clens), np.std(clens), np.max(clens), np.mean(qlens), np.std(qlens), np.max(qlens)


def normalize(data):
    ndata = []
    for dp in data:
        dp["title"] = u.normalize('NFKC', dp["title"]).replace(u"\xa0", u" ")
        for p in dp["paragraphs"]:
            p["context"] = u.normalize('NFKC', p["context"]).replace(u"\xa0", u" ")
            for qa in p["qas"]:
                qa["question"] = u.normalize('NFKC', qa["question"]).replace(u"\xa0", u" ")
                for a in qa["answers"]:
                    a["text"] = u.normalize('NFKC', a["text"]).replace(u"\xa0", u" ")
        ndata.append(dp)
    return ndata


def print_stats():
    train_paths = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.endswith(".train")]
    dev_paths = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.endswith(".dev")]
    test_paths = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.endswith(".test")]

    fmt_str = "{0:>10}: {1:>7.0f} {2:>7.2f} {3:>7.2f} {4:>7.0f} {5:>7.2f} {6:>7.2f} {7:>7.0f}"

    def loop(paths):
        for fpath in paths:
            with open(fpath, "r") as f:
                data = json.load(f)["data"]
            lang = fpath.split("/")[-1].split(".")[0]
            print(fmt_str.format(lang, *count(data)))

    print(
        "\n{0:>10}  {1:>7} {2:>7} {3:>7} {4:>7} {5:>7} {6:>7} {7:>7}".format(
            "", "# of Qs", "C mean", "C std", "C max", "Q mean", "Q std", "Q max"
        )
    )
    print("Train lang")
    loop(train_paths)
    print("\nDev lang")
    loop(dev_paths)
    print("\nTest lang")
    loop(test_paths)


def main():
    with open("./data/tydiqa/tydiqa-goldp-v1.1-train.json", "r") as f:
        data = json.load(f)
        version = data["version"]
        data = data["data"]
        data = normalize(data)
    langs = ["english", "arabic", "bengali", "finnish", "indonesian", "swahili", "korean", "russian", "telugu"]
    random.shuffle(langs)
    train_langs, test_langs = langs[:5], langs[5:]
    print(f"Train langs: {train_langs}")
    print(f"Test langs: {test_langs}")

    datasets = defaultdict(list)
    for dp in data:
        lang = dp["paragraphs"][0]["qas"][0]["id"].split("-")[0]
        if lang not in langs:
            raise ValueError("Datapoint does not have a valid id: {}".format(dp["id"]))
        datasets[lang].append(dp)
    assert len(data) == sum(len(ds) for ds in datasets.values())
    for ds in datasets.values():
        random.shuffle(ds)

    # write train and dev files
    os.makedirs(save_dir, exist_ok=True)
    for lang in train_langs:
        d = datasets[lang]
        split_idx = round(len(d) * 0.1)
        with open(os.path.join(save_dir, f"{lang}.dev"), "w") as f:
            json.dump({"version": version, "data": d[:split_idx]}, f)
        with open(os.path.join(save_dir, f"{lang}.train"), "w") as f:
            json.dump({"version": version, "data": d[split_idx:]}, f)

    # copy real dev files as test files
    for lang in test_langs:
        with open(test_path.format(lang), "r") as f:
            data = json.load(f)
            version = data["version"]
            data = data["data"]
            data = normalize(data)
        with open(os.path.join(save_dir, f"{lang}.test"), "w") as f:
            json.dump({"version": version, "data": data}, f)

    print_stats()


if __name__ == "__main__":
    main()
