# robust-maml

Code to reproduce the results described in the paper "Minimax and Neyman-Pearson Meta-Learning for Outlier Languages" [(aclweb)](https://aclanthology.org/2021.findings-acl.106/).

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Run data creation scripts (`make_pos_data.py` and `make_qa_data.py`. See top comments in files for more details)
5. Use the configs from the `configs` folder or suitably modify them as required to start training

To start training, simply pass the config path to `trainer.py`
```
$ python trainer.py --config_path=<config-path>
```

To run fast adapt, and evaluate the trained models on one outlier-language, run
```
$ python tester.py --test_lang=<lang> --model_path=<saved-model-path>
```

To run the tester on all outlier-languages, use `run_test.sh`. It contains two functions, `run_pos` and `run_qa` which will evaluate POS tagging and QA on all languages, respectively. The individual language results will be stored in `<saved-model-path>/result`. To merge them all into a single CSV file, run
```
$ python create_csv.py --model_path=<saved-model-path>
```

For QA, since we split the languages into two groups and train two models. Therefore, `create_csv.py` expects 2 paths: `--model_path` and `--model_path2`

### Citation
```
@inproceedings{ponti-etal-2021-minimax,
    title = "Minimax and Neyman{--}{P}earson Meta-Learning for Outlier Languages",
    author = "Ponti, Edoardo Maria  and
      Aralikatte, Rahul  and
      Shrivastava, Disha  and
      Reddy, Siva  and
      S{\o}gaard, Anders",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.106",
    pages = "1245--1260",
}
```
