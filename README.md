# A fully differentiable model for unsupervised singing voice separation

This is the source code for the experiments related to our work on a differentiable model for unsupervised singing voice separation.  

We proposed to extend the work of Schultze-Foster et al., and to build a complete, fully differentiable model by integrating a multipitch estimator and a novel differentiable voice assignment module within the core model.


__Note 1:__ This project builds upon the model of Schultze-Foster _et al._ and parts of the code are taken/adapted from their [repository](https://github.com/schufo/umss).

__Note 2:__ The trained models of [Cuesta _et al._](https://github.com/helenacuesta/multif0-estimation-polyvocals) (multiple-f0 estimation) and, [Cuesta and Gómez](https://github.com/helenacuesta/voas-vocal-quartets) (voice assignment) have been used in our experiments.

## Links

[:loud_sound: Audio examples](https://pierrechouteau.github.io/)

[:page_facing_up:]() [Schultze-Forster _et al._ paper](https://ieeexplore.ieee.org/document/10058592)

[:page_facing_up:]() [Multiple-f0 estimation paper](https://program.ismir2020.net/poster_2-18.html) | [Multiple-f0 Assignment paper](https://transactions.ismir.net/articles/10.5334/tismir.121)

[:file_folder:]() [CSD Database](https://zenodo.org/record/1286570#.Y0ZsbNJByUk) | [Cantoría Database](https://zenodo.org/record/5851070)


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
Note: 'TAG' is the evaluated model's name. (Example: UMSS_4s_bcbq)


## Inference

To separate the voices of a mixture, run the `inference.py` script:

```bash
python inference.py --audio_path AUDIO_PATH --tag TAG --mode MODE --output_dir OUTPUT_DIR --device DEVICE
```

with:
- `AUDIO_PATH`: path to the mixture audio file
- `TAG`: name of the model to use (between our trained models, default is `W-Up_bcbq`)
- `MODE`: mode to save the audio files (between `segmented_audio` and `full_audio`, default is `segmented_audio`).
- `OUTPUT_DIR`: path where the separated voices will be saved (default is `./output`)
- `DEVICE`: device to use (between `cpu` and `cuda`, default is `cpu`)


Note: Except for `AUDIO_PATH`, all other arguments are optional and have default values.


## Trained models

The trained models used in our experiments are available [here](https://drive.google.com/drive/folders/1OICrCIajHvA-gv7XofF5GWrmEp0ME3e9?usp=drive_link).
