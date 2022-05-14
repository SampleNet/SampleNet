# SampleNet Example Code

## Create and activate conda environment
```
conda create -n snet python=3.10
conda activate snet
```

## Install [pytorch 1.11.0](https://pytorch.org/get-started/locally/)
```
pip install torch==1.11.0 torchvision~=0.12.0 torchaudio~=0.11.0
```

## Install requirements
```
pip install -r requirements.txt
```

## Run code
```
python traffic.py
python weather.py
```
Change the `device` variable from `cuda:0` to `cpu` if needed in the scripts. CPU runs are very slow.

