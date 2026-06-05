
### Environment Configuration

```
conda create --name mutd python=3.8

source activate mutp

conda install --file requirements.txt -c pytorch
```


### Datasets Processing

```
python process_gdelt.py

python process_icews.py
```
This will create the files required to compute the filtered metrics.


### Reproducing results of MuTP

In order to reproduce the results of MuTP on three TKG datasets,  run the following commands:

```
python learner.py --dataset GDELT --rank 6000 --batch_size 2000 --learning_rate 0.2 --emb_reg 0.0001 --time_reg 0.01  --max_epochs 500 --valid_freq 50 --gpu 0 --scale_reg 0.05 --n_scales 4

python learner.py --dataset ICEWS14 --rank 6000 --batch_size 4000 --learning_rate 0.01 --emb_reg 0.01 --time_reg 0.01 --max_epochs 500 --valid_freq 10 --gpu 1 --scale_reg 0.05 --n_scales 3

python  learner.py --dataset ICEWS05-15 --rank 8000 --batch_size 3000 --learning_rate 0.008 --emb_reg 0.002 --time_reg 0.1 --max_epochs 500 --valid_freq 10 --gpu 2  --scale_reg 0.0005 --n_scales 4

```