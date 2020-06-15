# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download data: `python download_utils.py`
4. Train pos-tagger: `python simple-trainer.py`

*Note*: Model is currently overfitting badly. Need to tune the hyperparams.
