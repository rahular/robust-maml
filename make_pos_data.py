"""
To  download UD, do:
>> mkdir -p data/ud  && cd data/ud
>> curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz
>> tar -xf ud-treebanks-v2.6.tgz
>> python make_pos_data.py (execute this file)
"""

import os
import glob
import pyconll
import tqdm

from shutil import copyfile

# download UD and point to it here.
UD_DIR = "./data/ud/ud-treebanks-v2.6"
test_langs = set(
    [
        "abq",
        "bm",
        "bxr",
        "eu",
        "gun",
        "pcm",
        "th",
        "wbp",
        "id",
        "tl",
        "ta",
        "te",
        "wo",
        "yo",
    ]
)


def get_info(split, langs):
    count, empty = 0, 0
    paths = []
    for filename in glob.iglob(UD_DIR + "**/**", recursive=True):
        if ".conllu" in filename:
            lang = filename.split("/")[-1].split("_")[0]
            if lang in langs and split in filename:
                count += 1
                for ts in pyconll.iter_from_file(filename):
                    if len(ts) < 2:
                        continue
                    # if form == '_', it means that text is missing
                    if ts[0].form == ts[1].form == "_":
                        empty += 1
                        print(f"Text missing in {filename}...")
                    else:
                        paths.append(filename)
                    break
    return count, empty, paths


def save_files(paths, split):
    dst_dir = "./data/pos/all/"
    os.makedirs(dst_dir, exist_ok=True)
    for path in paths:
        src = path
        dst = dst_dir + path.split("/")[-1].split("-")[0] + f".{split}"
        copyfile(src, dst)


def main():
    all_langs = []
    for filename in glob.iglob(UD_DIR + "**/**", recursive=True):
        if ".conllu" in filename:
            lang = filename.split("/")[-1].split("_")[0]
            all_langs.append(lang)
    all_langs = set(all_langs)

    train_langs = all_langs.difference(test_langs)
    print("Train langs: ", len(train_langs))
    print("Test langs: ", len(test_langs))

    print("\nFinding train files...")
    train_count, empty_count, train_paths = get_info("train", train_langs)
    print(f"Found {empty_count}/{train_count} empty train files...")
    print(f"So we can train on {len(train_paths)} datasets!")

    print("\nFinding dev files...")
    dev_count, empty_count, dev_paths = get_info("dev", train_langs)
    print(f"Found {empty_count}/{dev_count} empty dev files...")
    print(f"So we can validate on {len(dev_paths)} datasets!")

    print("\nFinding test files...")
    test_count, empty_count, test_paths = get_info("test", test_langs)
    print(f"Found {empty_count}/{test_count} empty test files...")
    print(f"So we can test on {len(test_paths)} datasets!")

    print("\nSaving all files...")
    save_files(train_paths, "train")
    save_files(dev_paths, "dev")
    save_files(test_paths, "test")


if __name__ == "__main__":
    main()
