import os
import json

from urllib import request
from data_utils import get_data_config


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


if __name__ == "__main__":
    config = get_data_config()
    download_partut(config)
