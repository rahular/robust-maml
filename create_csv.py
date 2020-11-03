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


def main():
    args = init_args()
    result_path = os.path.join(args.model_path, "result")
    files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.endswith("json")]

    datasets, k0, k1, k2, k5, k10 = [], [], [], [], [], []
    for file_ in tqdm.tqdm(sorted(files)):
        with open(file_, "r") as f:
            data = json.load(f)
            datasets.append(file_.split("/")[-1].split(".")[0])
            k0.append(round(data["0"]["f1"] * 100, 2))
            k1.append(round(data["1"]["f"] * 100, 2))
            k2.append(round(data["2"]["f"] * 100, 2))
            k5.append(round(data["5"]["f"] * 100, 2))
            k10.append(round(data["10"]["f"] * 100, 2))

    df = pd.DataFrame()
    df["lang"] = datasets
    df["k=0"] = k0
    df["k=1"] = k1
    df["k=2"] = k2
    df["k=5"] = k5
    df["k=10"] = k10

    df.to_csv(os.path.join(result_path, "combined_result.csv"))


if __name__ == "__main__":
    main()
