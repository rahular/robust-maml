# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download UD: Use [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226) to download all treebanks, unzip and place them inside `data` folder
4. TODO: Make a script to create the `x` folder inside `data`
5. Create new configs in the `configs` folder and use them to start training

Trainer usage
```
usage: trainer.py [-h] --config_path CONFIG_PATH [--train_type {meta,mtl}]

Train a classifier

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path of the config containing training params
  --train_type {meta,mtl}
                        Whether to perform MTL or meta-training
```

Tester usage
```
usage: tester.py [-h] --test_path TEST_PATH --model_path MODEL_PATH
                 [-e {meta,full,both}]

Test POS tagging on various UD datasets

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        Datasets to test on
  --model_path MODEL_PATH
                        Path of the model to load
  -e {meta,full,both}, --eval_type {meta,full,both}
                        Type of evaluation to perform
```