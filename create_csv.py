import os
import json
import tqdm
import argparse

import pandas as pd


def init_args():
    parser = argparse.ArgumentParser(description="Compile results of a model into a .csv file")
    parser.add_argument(
        "--model_path", dest="model_path", type=str, help="Path of the model folder", required=True,
    )
    return parser.parse_args()


def for_seq_lbl(files, result_path):
    datasets, k0, k5, s5, k10, s10, k20, s20 = [], [], [], [], [], [], [], []
    for file_ in tqdm.tqdm(sorted(files)):
        with open(file_, "r") as f:
            data = json.load(f)
            datasets.append(file_.split("/")[-1].split(".")[0])
            k0.append(round(data["0"]["f1"] * 100, 2))
            k5.append(round(data["5"]["f"] * 100, 2))
            s5.append(round(data["5"]["f_stdev"] * 100, 2))
            k10.append(round(data["10"]["f"] * 100, 2))
            s10.append(round(data["10"]["f_stdev"] * 100, 2))
            k20.append(round(data["20"]["f"] * 100, 2))
            s20.append(round(data["20"]["f_stdev"] * 100, 2))
    df = pd.DataFrame()
    df["lang"] = datasets
    df["k=0"] = k0
    df["k=20"] = k20
    df["20_std"] = s20
    df["k=5"] = k5
    df["5_std"] = s5
    df["k=10"] = k10
    df["10_std"] = s10
    df.to_csv(os.path.join(result_path, "combined_result.csv"))


def for_qa(files, result_path):
    datasets, e0, f0, e5, f5, es5, fs5, e10, f10, es10, fs10, e20, f20, es20, fs20 = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for file_ in tqdm.tqdm(sorted(files)):
        with open(file_, "r") as f:
            data = json.load(f)
            datasets.append(file_.split("/")[-1].split(".")[0])
            e0.append(round(data["0"]["exact"], 2))
            f0.append(round(data["0"]["f1"], 2))

            e5.append(round(data["5"]["exact"], 2))
            f5.append(round(data["5"]["f1"], 2))
            es5.append(round(data["5"]["exact_stdev"], 2))
            fs5.append(round(data["5"]["f1_stdev"], 2))

            e10.append(round(data["10"]["exact"], 2))
            f10.append(round(data["10"]["f1"], 2))
            es10.append(round(data["10"]["exact_stdev"], 2))
            fs10.append(round(data["10"]["f1_stdev"], 2))

            e20.append(round(data["20"]["exact"], 2))
            f20.append(round(data["20"]["f1"], 2))
            es20.append(round(data["20"]["exact_stdev"], 2))
            fs20.append(round(data["20"]["f1_stdev"], 2))

    df = pd.DataFrame()
    df["lang"] = datasets
    df["e0"] = e0
    df["f0"] = f0
    df["e5"] = e5
    df["e5_stdev"] = es5
    df["f5"] = f5
    df["f5_stdev"] = fs5
    df["e10"] = e10
    df["e10_stdev"] = es10
    df["f10"] = f10
    df["f10_stdev"] = fs10
    df["e20"] = e20
    df["e20_stdev"] = es20
    df["f20"] = f20
    df["f20_stdev"] = fs20
    df.to_csv(os.path.join(result_path, "combined_result.csv"))


def main():
    args = init_args()
    result_path = os.path.join(args.model_path, "result")
    files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.endswith("json")]
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        data_dir = json.load(f)["data_dir"]
    if "tydiqa" in data_dir:
        for_qa(files, result_path)
    else:
        for_seq_lbl(files, result_path)



if __name__ == "__main__":
    main()
