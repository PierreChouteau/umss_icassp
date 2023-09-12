# A fully differentiable model for unsupervised singing voice separation

This is the source code for the experiments related to our work on a differentiable model for unsupervised singing voice separation.  

We proposed to extend the work of Schultze-Foster et al., and to build a complete, fully differentiable model by integrating a multipitch estimator and a novel differentiable voice assignment module within the core model.

## Links
[:loud_sound: Audio examples](https://pierrechouteau.github.io/)

[:page_facing_up: Schultze-Forster _et al._ Paper's](https://ieeexplore.ieee.org/document/10058592)

[:file_folder:]() [CSD Database](https://zenodo.org/record/1286570#.Y0ZsbNJByUk) | [Cantoría Database](https://zenodo.org/record/5851070)

[:microphone:]() [Multiple-f0 estimation](https://github.com/helenacuesta/multif0-estimation-polyvocals) | [Multiple-f0 Assignement](https://github.com/helenacuesta/voas-vocal-quartets)

## Requirements

The following packages are required:

    pytorch=1.6.0
    matplotlib=3.3.1
    python-sounddevice=0.4.0
    scipy=1.5.2
    torchaudio=0.6.0
    tqdm=4.49.0
    pysoundfile=0.10.3
    librosa=0.8.0
    scikit-learn=0.23.2
    tensorboard=2.3.0
    resampy=0.2.2
    pandas=1.2.3

These packages can be found using the conda-forge and pytorch channels.
Python 3.7 or 3.8 is recommended.
From a new conda environment:

```
conda update conda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --add channels pytorch
conda install pytorch=1.6.0
conda install numpy=1.23.5 matplotlib=3.3.1 python-sounddevice=0.4.0 scipy=1.5.2 torchaudio=0.6.0 tqdm=4.49.0 pysoundfile=0.10.3 librosa=0.8.0 scikit-learn=0.23.2 tensorboard=2.3.0 resampy=0.2.2 pandas=1.2.3 configargparse=0.13.0
pip install pumpp==0.6.0 nnAudio=0.3.2
```

or you can use the provided environment.yml file:

```
conda env create -f environment.yml
```
    
## Training

```
python train.py -c config.txt
```
``` 
python train_u_nets.py -c unet_config.txt
``` 
## Evaluation

```
python eval.py --tag 'TAG' --f0-from-mix --test-set 'CSD'
```
Note : 'TAG' is the evaluated model's name. (Example: unsupervised_2s_satb_bcbq_mf0_1)

