# Usage

## Genreate datasource
```
tar zxf atari_min.tar.gz
python create_atari.py
```

## Datasource Class
It is in the `data_atari.py` file, please check `train.py` to see the usage,
and you can also create your own clevr datasource too.

## Training with the datasource
```
python train.py --data-dir [PATH_TO_TRAIN.PT]
```
