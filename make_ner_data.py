"""
This code is adapted from  https://github.com/cambridgeltl/parameter-factorization/blob/master/tools/split_wikiann.py

To download WikiAnn, do:
>> mkdir -p data/wikiann && cd  data/wikiann
Download the tar files from https://drive.google.com/drive/folders/1Q-xdT99SeaCghihGa7nRkcXGwRGUIsKN
>> find data/wikiann -name '*.tar.gz' -execdir tar -xzvf '{}' \;
>> rm data/wikiann/*.tar.gz
>> python make_ner_data.py data/wikiann (execute this file)
>> rm data/wikiann/wikiann-*.bio
"""

import os
import sys
import math
import random

from shutil import copyfile

random.seed(313)
basedir = sys.argv[1]
# fmt: off
test_langs = set(
    ['ab', 'ady', 'av', 'ay', 'ba', 'bxr', 'ce', 'chr', 'chy', 'cr', 'eu', 'gn', 'ik', 'ik', 'ja', 'ka',
     'kbd', 'kl', 'km', 'kn', 'ko', 'lbe', 'lez', 'lo', 'ml', 'mo', 'nah', 'nv', 'qu', 'ta', 'te', 'th',
     'vi', 'xal', 'xmf', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue']
)
# fmt: on


def printout(split, outdir, sentences):
    fout = open(outdir + "." + split + ".bio", "w")
    for sent in sentences:
        for l in sent:
            fout.write(l)
        fout.write("\n")
    fout.close()


def split_data():
    for filename in os.listdir(basedir):
        if filename.endswith(".bio"):
            print("Splitting", filename)
            sentences = []
            current = []
            fin = open(os.path.join(basedir, filename))
            for l in fin:
                if not l.strip():
                    sentences.append(current)
                    current = []
                else:
                    current.append(l)
            if current:
                sentences.append(current)
            random.shuffle(sentences)
            if len(sentences) >= 10:
                print("Found {} sentences".format(len(sentences)))
                delim1 = int(math.floor(len(sentences) * 0.8))
                delim2 = int(math.floor(len(sentences) * 0.9))
                print(
                    "Split sizes: train {}, dev {}, test {}.".format(delim1, delim2 - delim1, len(sentences) - delim2)
                )
                os.mkdir(os.path.join(basedir, filename[8:-4]))
                outdir = os.path.join(basedir, filename[8:-4], filename[8:-4])
                printout("train", outdir, sentences[:delim1])
                printout("dev", outdir, sentences[delim1:delim2])
                printout("test", outdir, sentences[delim2:])


def copy(langs, split):
    dst_dir = "data/ner/all"
    os.makedirs(dst_dir, exist_ok=True)
    for lang in langs:
        src = os.path.join(basedir, lang, f"{lang}.{split}.bio")
        dst = os.path.join(dst_dir, f"{lang}.{split}")
        copyfile(src, dst)


def collect():
    all_langs = set([d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))])
    train_langs = all_langs.difference(test_langs)
    print("Train langs: ", len(train_langs))
    print("Test langs: ", len(test_langs))
    copy(train_langs, "train")
    copy(train_langs, "dev")
    copy(test_langs, "test")


def main():
    # split_data()
    collect()


if __name__ == "__main__":
    main()
