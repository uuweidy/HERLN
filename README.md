# HERLN
This is the code for paper **Hawkes based Representation Learning for Reasoning over Scale-free Community-structured Temporal Knowledge Graphs**.


## Data Preprocess
We use a software **Gephi** to get communities of nodes. See https://github.com/gephi/gephi. We also also provide the `train.csv` in these datasets.

## Train & Test
```
cd src

python main.py -d ICEWS14s --self-loop --layer-norm --weight 0.5 --theta 1  --entity-prediction --relation-prediction --gpu 0
```

Results will be saved in the folder `checkpoints`.
