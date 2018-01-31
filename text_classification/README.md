# Text Classification

## Data Preparation
```
wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
tar zxf aclImdb_v1.tar.gz
```

## Fluid Training
```
python train.py --dict_path 'aclImdb/imdb.vocab'
```
## V2 Training
```
python train_v2.py 'aclImdb/imdb.vocab'
```
