# A fully differentiable model for unsupervised singing voice separation

## Description

This is the source code for the experiments related to our work on a differentiable model for unsupervised singing voice separation.  

We proposed to extend the work of Schultze-Foster et al.^1^, and to build a complete, fully differentiable model by integrating a multipitch estimator and a novel differentiable voice assignment module within the core model.

__Note 1:__ This project builds upon the model of Schultze-Foster et al. and parts of the code are taken/adapted from their [repository](https://github.com/schufo/umss).

__Note 2:__ The trained models of [multif0-estimation-polyvocals](https://github.com/helenacuesta/multif0-estimation-polyvocals)^2^ and, [voas-vocal-quartets](https://github.com/helenacuesta/voas-vocal-quartets)^3^ have been used in our experiments.

1. K. Schulze-Forster, G. Richard, L. Kelley, C. S. J. Doire and R. Badeau, "Unsupervised Music Source Separation Using Differentiable Parametric Source Models," _IEEE/ACM Transactions on Audio, Speech, and Language Processing_, pp. 1-14, 2023

2. H. Cuesta, B. McFee, and E. Gómez, “Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks”, in _ISMIR_, Montréal, Canada, 2020

3. H. Cuesta and E. Gómez, “Voice Assignment in Vocal Quartets Using Deep Learning Models Based on Pitch Salience”, _Transactions of the International Society for Music Information_, 2022

## Links

[:loud_sound: Audio examples](https://pierrechouteau.github.io/)

[:file_folder:]() [CSD Database](https://zenodo.org/record/1286570#.Y0ZsbNJByUk) | [Cantoría Database](https://zenodo.org/record/5851070)


## Installing the working environment

### With conda

Create an environment using the `environment.yml` file:
```
conda env create -f environment.yml
```
    
## Training

To start the training, run the `train.py` or `train_unets.py` script:
```
python train.py -c config.txt
```

``` 
python train_u_nets.py -c unet_config.txt
```

## Evaluation

To evaluate the model, run the `eval.py` script:

```
python eval.py --tag 'TAG' --f0-from-mix --test-set 'CSD'
```
Note: 'TAG' is the evaluated model's name. (Example: UMSS_4s_bcbq)


## Trained models

The trained models used in our experiments are available [here](https://drive.google.com/drive/folders/1OICrCIajHvA-gv7XofF5GWrmEp0ME3e9?usp=drive_link).
