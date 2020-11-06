# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Run data creation scripts (`make_pos_data.py` and `make_ner_data.py`. See top comments in files for more details)
5. Copy configs from the `configs` folder and use them to start training (DO NOT modify existing configs)

Trainer usage
```
usage: trainer.py [-h] --config_path CONFIG_PATH [--train_type {meta,mtl}]

Train a classifier

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path of the config containing training params
```

Tester usage
```
usage: tester.py [-h] --test_lang TEST_LANG --model_path MODEL_PATH

Test POS tagging on various UD datasets

optional arguments:
  -h, --help            show this help message and exit
  --test_lang TEST_LANG
                        Language to test on
  --model_path MODEL_PATH
                        Path of the model to load
```

The test results will be stored under `<MODEL_PATH/result>`
