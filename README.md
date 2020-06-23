# pos-bert

1. Create conda env: `conda create -n pos-bert python=3.7`
2. Install dependencies: `pip install -r requirements.txt`
3. Download UD: Use [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226) to download all treebanks, unzip and place them inside `data` folder
4. Train pos-tagger: `python simple-trainer.py`

### Current results

Results after trained for *3* epochs:

|Model                         |Data  |Train F1|Test F1|
|------------------------------|------|--------|-------|
|bert-multilingual-base-cased  |partut|0.992   |0.803  |
|bert-multilingual-base-cased  |EWT   |?       |0.782  |
