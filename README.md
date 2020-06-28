# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download UD: Use [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226) to download all treebanks, unzip and place them inside `data` folder
4. Rename the files in the concerned treebanks to `train/dev/test.conllu`
5. Train pos-tagger: `python simple-trainer.py <dataset-name>`

### Current results

|Model                         |Data  |Train F1|Test F1|
|------------------------------|------|--------|-------|
|bert-base-cased               |partut|0.9876  |0.9605 |
|bert-multilingual-base-cased  |partut|0.9959  |0.9605 |
|bert-base-cased               |ewt   |0.9982  |0.9577 |
|bert-multilingual-base-cased  |ewt   |0.9978  |0.9485 |

