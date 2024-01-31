# A fully differentiable model for unsupervised singing voice separation

## Description

This is the source code for the experiments related to our work on a differentiable model for unsupervised singing voice separation.  

We proposed to extend the work of Schultze-Foster et al., and to build a complete, fully differentiable model by integrating a multipitch estimator and a novel differentiable voice assignment module within the core model.

__Note 1:__ This project builds upon the model of Schultze-Foster _et al._ and parts of the code are taken/adapted from their [repository](https://github.com/schufo/umss).

__Note 2:__ The trained models of [Cuesta _et al._](https://github.com/helenacuesta/multif0-estimation-polyvocals) (multiple-f0 estimation) and, [Cuesta and Gómez](https://github.com/helenacuesta/voas-vocal-quartets) (voice assignment) have been used in our experiments.

## Links

[:loud_sound: Audio examples](https://pierrechouteau.github.io/)

[:page_facing_up:]() [Schultze-Forster _et al._ paper](https://ieeexplore.ieee.org/document/10058592)

[:page_facing_up:]() [Multiple-f0 estimation paper](https://program.ismir2020.net/poster_2-18.html) | [Multiple-f0 Assignment paper](https://transactions.ismir.net/articles/10.5334/tismir.121)

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
