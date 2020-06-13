import os
import json

from urllib import request


def download_partut(config):
    if not os.path.exists(config["ParTut_path"]):
        os.makedirs(config["ParTut_path"])
    request.urlretrieve(
        config["url_ParTut_train"], os.path.join(config["ParTut_path"], "train.conllu")
    )
    request.urlretrieve(
        config["url_ParTut_dev"], os.path.join(config["ParTut_path"], "dev.conllu")
    )
    request.urlretrieve(
        config["url_ParTut_test"], os.path.join(config["ParTut_path"], "test.conllu")
    )


def get_config():
    with open("download_config.json", "r") as f:
        return json.load(f)


if __name__ == "__main__":
    config = get_config()
    download_partut(config)
