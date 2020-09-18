# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download UD: Use [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226) to download all treebanks, unzip and place them inside `data` folder
4. Rename the files in the concerned treebanks to `train/dev/test.conllu`
5. Meta-train pos-tagger: `python meta_trainer.py ewt partut` (can take arbitrary number of datasets)
6. Evaluate pos-tagger: `python test.py ewt partut --split=test`

