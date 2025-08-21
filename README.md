# Toy example of an Unet ML reconstruction of swath data to a gridded field.

This script will create a number of training samples, and validation samples. Train a simple Unet model and then plot 20 random validation samples.

## Environment

```shell
$ conda create -n OSSE_Unet python=3.11 xarray matplotlib
$ pip install torch # this doesn't appear to be on conda forge?
```

## Usage

```shell
$ python main.py # this will run with the config defaults
```

## Configuration

There are a number of options that can be specified

```shell
  --ny NY
  --nx NX
  --train-samples TRAIN_SAMPLES
  --val-samples VAL_SAMPLES
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --base-ch BASE_CH
  --seed SEED
  --swath-width SWATH_WIDTH
  --n-swaths N_SWATHS
  --swath-angle SWATH_ANGLE
  --swath-gap SWATH_GAP
  --scattered-frac SCATTERED_FRAC
  --noise-std NOISE_STD
```