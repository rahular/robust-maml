# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download data: `python download_utils.py`
4. Train pos-tagger: `python simple-trainer.py`

*Note*: Model is currently overfitting badly. Need to tune the hyperparams.

### Current results

|Model                         |Data  |Train F1|Test F1|
|------------------------------|------|--------|-------|
|bert-multilingual-base-cased  |partut|0.992   |0.363  |
|bert-multilingual-base-uncased|partut|0.990   |0.556  |
|bert-base-cased               |partut|0.988   |0.470  |
|bert-base-uncased             |partut|0.988   |0.519  |
