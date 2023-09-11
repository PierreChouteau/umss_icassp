from pathlib import Path
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import scipy.special

import glob
import argparse
import random
import torch
import torchaudio
import tqdm
import os
import pickle
import csv
import itertools

import utils
import ddsp.core

import librosa
from librosa.util.utils import fix_length

import pumpp
import matplotlib.pyplot as plt

import models
from nnAudio import features


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    if args.dataset == 'musdb':
        parser.add_argument('--is-wav', action='store_true', default=False,
                            help='loads wav instead of STEMS')
        parser.add_argument('--samples-per-track', type=int, default=64)
        parser.add_argument(
            '--source-augmentations', type=str, nargs='+',
            default=['gain', 'channelswap']
        )

        args = parser.parse_args()
        dataset_kwargs = {
            'root': args.root,
            'is_wav': args.is_wav,
            'subsets': 'train',
            'target': args.target,
            'download': args.root is None,
            'seed': args.seed
        }

        source_augmentations = Compose(
            [globals()['_augment_' + aug] for aug in args.source_augmentations]
        )

        train_dataset = MUSDBDataset(
            split='train',
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            **dataset_kwargs
        )

        valid_dataset = MUSDBDataset(
            split='valid', samples_per_track=1, seq_duration=None,
            **dataset_kwargs
        )


    elif args.dataset == 'CSD':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--train-song', type=str, default='Nino Dios', choices=['El Rossinyol', 'Locus Iste', 'Nino Dios'])
        parser.add_argument('--val-song', type=str, default='Locus Iste', choices=['El Rossinyol', 'Locus Iste', 'Nino Dios'])
        parser.add_argument('--f0-cuesta', action='store_true', default=False)

        args = parser.parse_args()

        train_dataset = CSD(song_name=args.train_song,
                            conf_threshold=args.confidence_threshold,
                            example_length=args.example_length,
                            allowed_voices=args.voices,
                            n_sources=args.n_sources,
                            # singer_nb=[2,3,4],
                            singer_nb=[2],
                            random_mixes=True,
                            f0_from_mix=args.f0_cuesta)

        valid_dataset = CSD(song_name=args.val_song,
                            conf_threshold=args.confidence_threshold,
                            example_length=args.example_length,
                            allowed_voices=args.voices,
                            n_sources=args.n_sources,
                            # singer_nb=[2,3,4],
                            singer_nb=[3],
                            random_mixes=False,
                            f0_from_mix=args.f0_cuesta)

    elif args.dataset == 'BCBQ':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--f0-cuesta', action='store_true', default=False)
        args = parser.parse_args()

        if args.one_song:
            train_dataset = BCBQDataSets(data_set='BC',
                                         validation_subset=False,
                                         conf_threshold=args.confidence_threshold,
                                         example_length=args.example_length,
                                         n_sources=args.n_sources,
                                         random_mixes=True,
                                         return_name=False,
                                         allowed_voices=args.voices,
                                         f0_from_mix=args.f0_cuesta,
                                         cunet_original=args.original_cu_net,
                                         one_song=True,
                                         F0_models=args.F0_models,
                                         F0_models_trainable=args.F0_models_trainable)

            valid_dataset = BCBQDataSets(data_set='BC',
                                         validation_subset=True,
                                         conf_threshold=args.confidence_threshold,
                                         example_length=args.example_length,
                                         n_sources=args.n_sources,
                                         random_mixes=False,
                                         return_name=False,
                                         allowed_voices=args.voices,
                                         f0_from_mix=args.f0_cuesta,
                                         cunet_original=args.original_cu_net,
                                         one_song=True,
                                         F0_models=args.F0_models,
                                         F0_models_trainable=args.F0_models_trainable)
        else:
            bc_train = BCBQDataSets(data_set='BC',
                                    validation_subset=False,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=True,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net,
                                    F0_models=args.F0_models,
                                    F0_models_trainable=args.F0_models_trainable)

            bq_train = BCBQDataSets(data_set='BQ',
                                    validation_subset=False,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=True,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net,
                                    F0_models=args.F0_models,
                                    F0_models_trainable=args.F0_models_trainable)

            bc_val = BCBQDataSets(data_set='BC',
                                  validation_subset=True,
                                  conf_threshold=args.confidence_threshold,
                                  example_length=args.example_length,
                                  n_sources=args.n_sources,
                                  random_mixes=False,
                                  return_name=False,
                                  allowed_voices=args.voices,
                                  f0_from_mix=args.f0_cuesta,
                                  cunet_original=args.original_cu_net,
                                  F0_models=args.F0_models,
                                  F0_models_trainable=args.F0_models_trainable)

            bq_val = BCBQDataSets(data_set='BQ',
                                  validation_subset=True,
                                  conf_threshold=args.confidence_threshold,
                                  example_length=args.example_length,
                                  n_sources=args.n_sources,
                                  random_mixes=False,
                                  return_name=False,
                                  allowed_voices=args.voices,
                                  f0_from_mix=args.f0_cuesta,
                                  cunet_original=args.original_cu_net,
                                  F0_models=args.F0_models,
                                  F0_models_trainable=args.F0_models_trainable)

            train_dataset = torch.utils.data.ConcatDataset([bc_train, bq_train])
            valid_dataset = torch.utils.data.ConcatDataset([bc_val, bq_val])
            
    elif args.dataset == 'Cantoria':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--train-song', type=str, default='EJB1', choices=['EJB2', 'EJB1', 'CEA', 'HCB', 'LBM1', 'LBM2', 'LJT1', 'LJT2', 'LNG', 'RRC', 'SSS', 'THM', 'VBP', 'YSM'])
        parser.add_argument('--val-song', type=str, default='EJB2', choices=['EJB2', 'EJB1', 'CEA', 'HCB', 'LBM1', 'LBM2', 'LJT1', 'LJT2', 'LNG', 'RRC', 'SSS', 'THM', 'VBP', 'YSM'])
        parser.add_argument('--f0-cuesta', action='store_true', default=False)
        args = parser.parse_args()
        
        train_dataset = CantoriaDataSets(song_name=args.train_song,
                                        conf_threshold=args.confidence_threshold, 
                                        example_length=args.example_length, 
                                        allowed_voices=args.voices,
                                        return_name=False, 
                                        n_sources=args.n_sources, 
                                        random_mixes=True, 
                                        f0_from_mix=args.f0_cuesta, 
                                        cunet_original=args.original_cu_net, 
                                        F0_models=args.F0_models, 
                                        F0_models_trainable=args.F0_models_trainable,
                                        )

        valid_dataset = CantoriaDataSets(song_name=args.val_song,
                                        conf_threshold=args.confidence_threshold, 
                                        example_length=args.example_length, 
                                        allowed_voices=args.voices,
                                        return_name=False, 
                                        n_sources=args.n_sources, 
                                        random_mixes=False, 
                                        f0_from_mix=args.f0_cuesta, 
                                        cunet_original=args.original_cu_net, 
                                        F0_models=args.F0_models, 
                                        F0_models_trainable=args.F0_models_trainable,
        )
        
    elif args.dataset == 'String':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--f0-cuesta', action='store_true', default=False)
        args = parser.parse_args()

        train_dataset = String(data_set='String',
                                    validation_subset=False,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=False,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net,
                                    F0_models=args.F0_models,
                                    F0_models_trainable=args.F0_models_trainable)

        valid_dataset = String(data_set='String',
                                    validation_subset=True,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=False,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net,
                                    F0_models=args.F0_models,
                                    F0_models_trainable=args.F0_models_trainable)
        
    return train_dataset, valid_dataset, args



class CSD(torch.utils.data.Dataset):

    def __init__(self, song_name: str, conf_threshold=0.4, example_length=64000, allowed_voices='satb',
                 return_name=False, n_sources=2, singer_nb=[2, 3, 4], random_mixes=False, f0_from_mix=False,
                 plus_one_f0_frame=False, cunet_original=False):
        """

        Args:
            song_name: str, must be one of ['El Rossinyol', 'Locus Iste', 'Nino Dios']
            conf_threshold: float, threshold on CREPE confidence value to differentiate between voiced/unvoiced frames
            example_length: int, length of the audio examples in samples
            return_name: if True, the names of the audio examples are returned (composed of title and singer_ids)
            n_sources: int, number of source to be mixed, must be in [1, 4]
            singer_nb: list of int in [1, 2, 3, 4], numbers that specify which singers should be used,
                e.g. singer 2, singer 3, and singer 4. The selection is valid for each voice group (SATB)
            random_mixes: bool, if True, time-sections, singers, and voices will be randomly chosen each epoch
                as means of data augmentation (should only be used for training data). If False, a deterministic
                set of mixes is provided that is the same at every call (for validation and test data).
        """

        self.song_name = song_name
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.return_name = return_name
        self.n_sources = n_sources
        self.sample_rate = 16000
        self.singer_nb = singer_nb
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.plus_one_f0_frame=plus_one_f0_frame  # for NMF
        self.cunet_original = cunet_original  # add some f0 frames to match representation in U-Net

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]

        # song length in seconds
        if song_name == 'El Rossinyol': self.total_audio_length = 134; self.voice_ids = ['Soprano', 'ContraAlt','Tenor', 'Bajos']
        elif song_name == 'Locus Iste': self.total_audio_length = 190; self.voice_ids = ['Soprano', 'ContraAlt','tenor', 'Bajos']
        elif song_name == 'Nino Dios': self.total_audio_length = 103; self.voice_ids = ['Soprano', 'ContraAlt','tenor', 'Bajos']

        if song_name == 'El Rossinyol' : song_name = 'El_Rossinyol'
        elif song_name == 'Locus Iste' : song_name = 'Locus_Iste'
        elif song_name == 'Nino Dios' : song_name = 'Nino_Dios'

        self.audio_files = sorted(glob.glob('./Datasets/ChoralSingingDataset/{}/audio_16kHz/*.wav'.format(song_name)))
        self.crepe_dir = './Datasets/ChoralSingingDataset/{}/crepe_f0_center'.format(song_name)

        f0_cuesta_dir = './Datasets/ChoralSingingDataset/{}/mixtures_{}_sources/mf0_cuesta_processed/*.pt'.format(song_name, n_sources)
        self.f0_cuesta_files = sorted(list(glob.glob(f0_cuesta_dir)))

        if not random_mixes:
            # number of non-overlapping excerpts
            n_excerpts = self.total_audio_length * self.sample_rate // self.example_length
            excerpt_idx = [i for i in range(1, n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # possible combinations of singers
            singer_combinations = list(itertools.combinations_with_replacement([idx - 1 for idx in singer_nb], r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations, singer_combinations))
            self.n_examples = len(self.examples)

    def __len__(self):
        if self.random_mixes: return 1600
        else: return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for idx in self.voice_choices: probabilities[idx] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample a number of singer_nbs with replacement
            probabilities = torch.ones((4,))
            singer_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=True)

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (self.total_audio_length-self.example_length/self.sample_rate)

        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices), (tuple of singer ids))
            params = self.examples[idx]

            excerpt_idx, voice_indices, singer_indices = params
            audio_start_seconds = excerpt_idx * self.example_length / self.sample_rate

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)

        if self.plus_one_f0_frame: crepe_end_frame += 1

        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = self.song_name.replace(' ', '_').lower()
        contained_singer_ids = []

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]

            audio_file = [f for f in self.audio_files if voice in f][singer_indices[n]]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            if not self.f0_from_mix:
                confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
                confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
                f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
                frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
                frequency = np.where(confidence < self.conf_threshold, 0, frequency)

                frequency = torch.from_numpy(frequency).type(torch.float32)
                f0_list.append(frequency)
                
                if not self.plus_one_f0_frame and not self.cunet_original:
                    assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'

            singer_id = '_' + voice[0].replace('C', 'A') + file_name[-6:]
            contained_singer_ids.append(singer_id)
            name += '{}'.format(singer_id)

        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            permutations = list(itertools.permutations(contained_singer_ids))
            permuted_mix_ids = [''.join(s) for s in permutations]
            f0_from_mix_file = [file for file in self.f0_cuesta_files if any([ids in file for ids in permuted_mix_ids])][0]
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])

        if self.return_name: return mix, frequencies, sources, name, voices
        else: return mix, frequencies, sources



class BCBQDataSets(torch.utils.data.Dataset):

    def __init__(self, data_set='BC', validation_subset=False, conf_threshold=0.4, example_length=64000, allowed_voices='satb',
                 return_name=False, n_sources=2, random_mixes=False, f0_from_mix=True, cunet_original=False, one_song=False, F0_models=False, F0_models_trainable=False):

        super().__init__()

        self.data_set = data_set
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.allowed_voices = allowed_voices
        self.return_name = return_name
        self.n_sources = n_sources
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.sample_rate = 16000
        self.cunet_original = cunet_original # if True, add 2 f0 values at start and end to match frame number in U-Net
        self.one_song = one_song
        self.F0_models = F0_models
        self.F0_models_trainable = F0_models_trainable

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                 'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]
        self.voice_ids = ['s', 'a', 't', 'b']

        self.audio_files = sorted(glob.glob('./Datasets/{}/audio_16kHz/*.wav'.format(data_set)))
        # file 17_BC021_part11_s_1ch.wav is empty  --> exclude 17_BC021_part11
        self.audio_files = [f for f in self.audio_files if '17_BC021_part11' not in f]
        self.crepe_dir = './Datasets/{}/crepe_f0_center'.format(data_set)

        if data_set == 'BC':
            if one_song:
                #  use BC 1 for training and BC 2 for validation
                if validation_subset: self.audio_files = [f for f in self.audio_files if '2_BC002' in f]
                else: self.audio_files = [f for f in self.audio_files if '1_BC001' in f]
            else:
                if validation_subset: self.audio_files = self.audio_files[- (14+17)*4 :]  # only 8_BC and 9_BC
                else: self.audio_files = self.audio_files[: - (14+17)*4]  # all except 8_BC and 9_BC
        elif data_set == 'BQ':
            if one_song:
                raise NotImplementedError
            else:
                if validation_subset: self.audio_files = self.audio_files[- (13+11)*4 :]  # only 8_BQ and 9_BQ
                else: self.audio_files = self.audio_files[: - (13+11)*4]  # all except 8_BQ and 9_BQ

        self.f0_cuesta_dir = './Datasets/{}/mixtures_{}_sources/mf0_cuesta_processed'.format(data_set, n_sources)

        if not random_mixes:
            # number of non-overlapping excerpts
            n_wav_files = len(self.audio_files)

            self.excerpts_per_part = ((10 * self.sample_rate) // self.example_length)

            n_excerpts = (n_wav_files // 4) * self.excerpts_per_part
            excerpt_idx = [i for i in range(n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations))
            self.n_examples = len(self.examples)

        else:
            # 1 epoch = going 4 times through parts and sample different voice combinations
            self.n_examples = len(self.audio_files) // 4 * 4
            if one_song: self.n_examples = self.n_examples * 3

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for i in self.voice_choices: probabilities[i] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
                voice_indices = sorted(voice_indices.tolist())
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (10 - 0.08 - self.example_length/self.sample_rate) + 0.04
            audio_start_seconds = audio_start_seconds.numpy()[0]
            if self.one_song:
                song_part_id = self.audio_files[idx * 4 // 12].split('/')[-1][:-10]
            else:
                song_part_id = self.audio_files[idx * 4 // 4].split('/')[-1][:-10]
        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices))
            params = self.examples[idx]

            excerpt_idx, voice_indices = params

            song_part_id = self.audio_files[excerpt_idx//self.excerpts_per_part * 4].split('/')[-1][:-10]

            audio_start_seconds = (excerpt_idx % self.excerpts_per_part) * self.example_length / self.sample_rate + 0.04

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)

        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = song_part_id + '_'

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]
            voice = '_' + voice + '_'

            audio_file = [f for f in self.audio_files if song_part_id in f and voice in f][0]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            if not self.f0_from_mix:
                confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
                confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
                f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
                frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
                frequency = np.where(confidence < self.conf_threshold, 0, frequency)

                frequency = torch.from_numpy(frequency).type(torch.float32)
                f0_list.append(frequency)
                
                # move to solve the frequency reference problem
                if not self.cunet_original:
                    assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'
                
            name += voice[1]


        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            f0_from_mix_file = self.f0_cuesta_dir + '/' + name + '.pt'
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])
                        
        if self.return_name: return mix, frequencies, sources, name, voices
        elif self.F0_models and not self.F0_models_trainable: return mix, frequencies, sources
        elif self.F0_models_trainable: return mix, frequencies, sources
        else: return mix, frequencies, sources
        
        
        
class CantoriaDataSets(torch.utils.data.Dataset):

    def __init__(self, 
                 song_name: str,
                 conf_threshold=0.4, 
                 example_length=64000, 
                 allowed_voices='satb',
                 return_name=False, 
                 n_sources=2, 
                 random_mixes=False, 
                 f0_from_mix=True, 
                 cunet_original=False, 
                 one_song=False,
                 F0_models=False, 
                 F0_models_trainable=False,
                 ):

        super().__init__()

        self.song_name = song_name
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.allowed_voices = allowed_voices
        self.return_name = return_name
        self.n_sources = n_sources
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.sample_rate = 16000
        self.cunet_original = cunet_original # if True, add 2 f0 values at start and end to match frame number in U-Net
        self.one_song = one_song
        self.F0_models = F0_models
        self.F0_models_trainable = F0_models_trainable

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                 'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]
        self.voice_ids = ['s', 'a', 't', 'b']

        if song_name == 'EJB1': self.total_audio_length = 300
        elif song_name == 'CEA': self.total_audio_length = 118
        elif song_name == 'EJB2': self.total_audio_length = 57
        elif song_name == 'HCB': self.total_audio_length = 36
        elif song_name == 'LBM1': self.total_audio_length = 300
        elif song_name == 'LBM2': self.total_audio_length = 271
        elif song_name == 'LJT1': self.total_audio_length = 300
        elif song_name == 'LJT2': self.total_audio_length = 205
        elif song_name == 'LNG': self.total_audio_length = 183
        elif song_name == 'RRC': self.total_audio_length = 56
        elif song_name == 'SSS': self.total_audio_length = 118
        elif song_name == 'THM': self.total_audio_length = 126
        elif song_name == 'VBP': self.total_audio_length = 69
        elif song_name == 'YSM': self.total_audio_length = 33
        
        self.audio_files = sorted(glob.glob('./Datasets/CantoriaDataset/{}/audio_16kHz/*.wav'.format(song_name)))
        self.crepe_dir = './Datasets/CantoriaDataset/{}/crepe_f0_center'.format(song_name)

        self.f0_cuesta_dir = './Datasets/CantoriaDataset/{}/mixtures_{}_sources/mf0_cuesta_processed'.format(song_name, n_sources)
        
        if not random_mixes:
            # number of non-overlapping excerpts
            n_excerpts = self.total_audio_length * self.sample_rate // self.example_length
            excerpt_idx = [i for i in range(1, n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations))
            self.n_examples = len(self.examples)


    def __len__(self):
        if self.random_mixes: return 400
        else: return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for i in self.voice_choices: probabilities[i] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
                voice_indices = sorted(voice_indices.tolist())
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample a number of singer_nbs with replacement
            probabilities = torch.ones((4,))

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (self.total_audio_length - self.example_length / self.sample_rate)
        
        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices), (tuple of singer ids))
            params = self.examples[idx]

            excerpt_idx, voice_indices = params
            audio_start_seconds = excerpt_idx * self.example_length / self.sample_rate

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)
        
        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2
            
            if idx == 73:
                crepe_start_frame -= 1
                crepe_end_frame += 1

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = self.song_name  + '_'

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]
            voice = '_' + voice

            audio_file = [f for f in self.audio_files if voice in f][0]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            if not self.f0_from_mix:
                confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
                confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
                f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
                frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
                frequency = np.where(confidence < self.conf_threshold, 0, frequency)

                frequency = torch.from_numpy(frequency).type(torch.float32)
                f0_list.append(frequency)
                
                # move to solve the frequency reference problem
                if not self.cunet_original:
                    assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'
                
            name += voice[1]


        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            f0_from_mix_file = self.f0_cuesta_dir + '/' + 'Cantoria_' + name + '.pt'
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])
                                                
        if self.return_name: return mix, frequencies, sources, name, voices
        elif self.F0_models and not self.F0_models_trainable: return mix, frequencies, sources
        elif self.F0_models_trainable: return mix, frequencies, sources
        else: return mix, frequencies, sources


class String(torch.utils.data.Dataset):

    def __init__(self, data_set='String', 
                 validation_subset=False, 
                 conf_threshold=0.4, 
                 example_length=64000, 
                 allowed_voices='satb',
                 return_name=False, 
                 n_sources=2, 
                 random_mixes=False, 
                 f0_from_mix=True, 
                 cunet_original=False, 
                 F0_models=False, 
                 F0_models_trainable=False,
                 ):

        super().__init__()

        self.data_set = data_set
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.allowed_voices = allowed_voices
        self.return_name = return_name
        self.n_sources = n_sources
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.sample_rate = 16000
        self.cunet_original = cunet_original # if True, add 2 f0 values at start and end to match frame number in U-Net
        self.F0_models = F0_models
        self.F0_models_trainable = F0_models_trainable

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                 'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]
        self.voice_ids = ['s', 'a', 't', 'b']

        self.audio_files = sorted(glob.glob('./Datasets/{}/audio_16kHz/*.wav'.format(data_set)))
        self.crepe_dir = './Datasets/{}/crepe_f0_center'.format(data_set)

        if validation_subset: self.audio_files = [f for f in self.audio_files if 'Violon2' in f]
        else: self.audio_files = [f for f in self.audio_files if ('Violon1', 'Viola', 'Cello') in f]

        self.f0_cuesta_dir = './Datasets/{}/mixtures_{}_sources/mf0_cuesta_processed'.format(data_set, n_sources)

        if not random_mixes:
            # number of non-overlapping excerpts
            n_wav_files = len(self.audio_files)

            self.excerpts_per_part = ((10 * self.sample_rate) // self.example_length)

            n_excerpts = (n_wav_files // 4) * self.excerpts_per_part
            excerpt_idx = [i for i in range(n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations))
            self.n_examples = len(self.examples)

        else:
            # 1 epoch = going 4 times through parts and sample different voice combinations
            self.n_examples = len(self.audio_files) // 4 * 4

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for i in self.voice_choices: probabilities[i] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
                voice_indices = sorted(voice_indices.tolist())
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (10 - 0.08 - self.example_length/self.sample_rate) + 0.04
            audio_start_seconds = audio_start_seconds.numpy()[0]
            if self.one_song:
                song_part_id = self.audio_files[idx * 4 // 12].split('/')[-1][:-10]
            else:
                song_part_id = self.audio_files[idx * 4 // 4].split('/')[-1][:-10]
        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices))
            params = self.examples[idx]

            excerpt_idx, voice_indices = params

            song_part_id = self.audio_files[excerpt_idx//self.excerpts_per_part * 4].split('/')[-1][:-10]

            audio_start_seconds = (excerpt_idx % self.excerpts_per_part) * self.example_length / self.sample_rate + 0.04

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)

        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = song_part_id + '_'

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]
            voice = '_' + voice + '_'

            audio_file = [f for f in self.audio_files if song_part_id in f and voice in f][0]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            if not self.f0_from_mix:
                confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
                confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
                f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
                frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
                frequency = np.where(confidence < self.conf_threshold, 0, frequency)

                frequency = torch.from_numpy(frequency).type(torch.float32)
                f0_list.append(frequency)
                
                # move to solve the frequency reference problem
                if not self.cunet_original:
                    assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'
                
            name += voice[1]


        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            f0_from_mix_file = self.f0_cuesta_dir + '/' + name + '.pt'
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])
                        
        if self.return_name: return mix, frequencies, sources, name, voices
        elif self.F0_models and not self.F0_models_trainable: return mix, frequencies, sources
        elif self.F0_models_trainable: return mix, frequencies, sources
        else: return mix, frequencies, sources


# -------- HCQT Computation -------------------------------------------------------------------------------------------
def get_hcqt_params():
    bins_per_octave = 60
    n_octaves = 6
    over_sample = 5
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256

    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample


def get_freq_grid():
    """
        Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _, over_sample) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        n_octaves * 12 * over_sample, f_min, bins_per_octave=bins_per_octave)
    return freq_grid


def get_time_grid(n_time_frames):
    """
        Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length, _) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """
        Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def hcqt_torch(audio):
    """
        Compute the harmonic CQT of a given audio signal.
        This function is a wrapper around the librosa implementation of the HCQT.
    """
    (
        bins_per_octave,
        n_octaves,
        harmonics,
        sr,
        f_min,
        hop_length,
        over_sample,
    ) = get_hcqt_params()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resample = torchaudio.transforms.Resample(16000, sr).to(device)
    
    resample = resample.to(device)
    audio = audio.to(device)
    
    audio = resample(audio)
    
    # Voir la fonction HCQT de pump, pour comprendre le raisonement
    ###
    samples = int(audio.size(1))
    time_to_frame = np.floor(samples // hop_length)
    n_frames = int(time_to_frame)
    ####
    
    mags = torch.empty((audio.size(0), len(harmonics), bins_per_octave*n_octaves, n_frames))
    dphases = torch.empty((audio.size(0), len(harmonics), bins_per_octave*n_octaves, n_frames))
        
    for h in harmonics:
        cqt_torch = features.cqt.CQT2010(sr=sr, 
                                        hop_length=hop_length, 
                                        fmin=f_min*h, 
                                        fmax=None, 
                                        n_bins=n_octaves * 12 * over_sample, 
                                        bins_per_octave=bins_per_octave,
                                        norm=True, 
                                        basis_norm=1, 
                                        window='hann', 
                                        pad_mode='reflect',
                                        trainable_STFT=False,
                                        filter_scale=1, 
                                        trainable_CQT=False,
                                        output_format='Complex', 
                                        earlydownsample=True, 
                                        verbose=False).to(device)
    
        # perform CQT
        audio_cqt = cqt_torch(audio)
        
        # partie imaginaire en 0, partie réelle en 1
        audio_cqt = audio_cqt[:, :, :, 1] + 1j * audio_cqt[:, :, :, 0]
        audio_cqt = fix_length(audio_cqt, n_frames)        
        
        # Transormation en magnitude et phase 
        mag = torch.abs(audio_cqt)
        mag = librosa.amplitude_to_db(mag.cpu(), ref=np.max) # TODO: Replace with torchaudio.transforms.AmplitudeToDB
        mags[:, h-1, :, :] = torch.tensor(mag)
        
        audio_cqt = audio_cqt.cpu()
        phase = torch.exp(1.0j * torch.angle(audio_cqt))
        phase = torch.angle(phase)
        
        # Transormation de la phase
        phase = np.transpose(phase, (0, 2, 1))
        
        dphase = np.empty(phase.shape, dtype='float32')
        zero_idx = [slice(None)] * phase.ndim
        zero_idx[1] = slice(1)
        else_idx = [slice(None)] * phase.ndim
        else_idx[1] = slice(1, None)
        zero_idx = tuple(zero_idx)
        else_idx = tuple(else_idx)
        dphase[zero_idx] = phase[zero_idx]
        dphase[else_idx] = np.diff(np.unwrap(phase, axis=1), axis=1)
        dphase = np.transpose(dphase, (0, 2, 1))
            
        dphases[:, h-1, :, :] = torch.tensor(dphase)
            
    return mags, dphases


def create_pump_object():
    (
        bins_per_octave,
        n_octaves,
        harmonics,
        sr,
        f_min,
        hop_length,
        over_sample,
    ) = get_hcqt_params()

    p_phdif = pumpp.feature.HCQTPhaseDiff(
        name="dphase",
        sr=sr,
        hop_length=hop_length,
        fmin=f_min,
        n_octaves=n_octaves,
        over_sample=over_sample,
        harmonics=harmonics,
        log=True,
    )

    pump = pumpp.Pump(p_phdif)

    return pump


def compute_pump_features(pump, audio_fpath):
    data = pump(audio_fpath)
    return data


def compute_pump_features_from_mix(pump, mix):
    y = librosa.resample(mix, 16000, 22050)
    data = pump(y=y, sr=22050)
    return data


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """
        Convert a pitch activation map to multif0 by thresholding peak values
        at thresh
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs, peak_thresh_mat


def mf0_assigned_to_salience_map(len_times, mf0_freqs):
    """
        Assign a multif0 array to a salience map by finding the nearest salience
        bin to each frequency in the multif0
    """
    freq_grid = get_freq_grid()
    freq_grid_int = np.array(freq_grid).astype(int)
    
    times = get_time_grid(len_times)

    salience_map = np.zeros((len(freq_grid_int), len(times)))
    freq_bins = np.zeros(salience_map.shape)
        
    for id_time, freq in enumerate(mf0_freqs):
        if freq != 0:
            id_freq = np.argwhere(freq_grid_int == int(freq))
            salience_map[id_freq, id_time] = 1.0 

        freq_bins[:, id_time] = freq_grid
        
    return salience_map, freq_bins


def mf0_assigned_to_salience_map_batch(mf0_predict, salience_shape):
    """ Permet de reconstruire un batch de salience map à partir d'un batch prédictions de mf0

    Args:
        mf0_predict (array): Prédiction des mf0 (shape : (batch, time_bins, n_voice))
        salience_shape (tuple): Shape de la salience map

    Returns:
        salience_maps: retourne le batch de salience maps reconstruit à partir des mf0 (shape : (batch, freq_bins, time_bins))
    """
    
    batch, len_times = mf0_predict.shape
    salience_maps_rec = np.zeros((salience_shape[0], salience_shape[2], salience_shape[3]))
    freq_bins_batch = np.zeros((salience_shape[0], salience_shape[2], salience_shape[3]))
    
    for id_batch in range(batch):
        mf0_freqs = mf0_predict[id_batch]
        salience_maps_rec[id_batch,:,:], freq_bins_batch[id_batch, :, :] = mf0_assigned_to_salience_map(len_times, mf0_freqs)

    return salience_maps_rec, freq_bins_batch


def pitch_activations_to_mf0_torch(pitch_activation_mat_torch, thresh):
    """
        Convert a pitch activation map to multif0 by thresholding peak values
        at thresh
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat_torch.size(1))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    peak_thresh_mat = torch.zeros(pitch_activation_mat_torch.size()).to(device)
    peaks = scipy.signal.argrelmax(pitch_activation_mat_torch.detach().cpu().numpy())
    peak_thresh_mat[peaks] = pitch_activation_mat_torch[peaks]

    est_freqs = [torch.tensor([], requires_grad=True) for _ in range(len(times))]
    for t in range(len(times)):
        idx = torch.where(peak_thresh_mat[:, t] >= thresh)
        freqs_t = freqs[idx[0]]
        est_freqs[t] = freqs_t

    return times, est_freqs



def save_multif0_output(times, freqs, output_path):
    """
        Save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)

def test_hcqt():
    audio_fpath = "/home/pierre/OneDrive/TELECOM/code/umss-pre/Datasets/ChoralSingingDataset/El_Rossinyol/audio_16kHz/rossinyol_Bajos_107.wav"
    pump = create_pump_object()
    features = compute_pump_features(pump, audio_fpath)

    hcqt = features['dphase/mag'][0]
    dphase = features['dphase/dphase'][0]
    
    hcqt = hcqt.transpose(2, 1, 0)[np.newaxis, :, :, :]
    dphase = dphase.transpose(2, 1, 0)[np.newaxis, :, :, :]
    
    print(hcqt.shape, dphase.shape)
    
    for i in range(5):
        plt.imshow(
            hcqt[0, i, :, :],
            origin="lower",
            aspect="auto",
            cmap="inferno",
        )
        plt.savefig(f"figures/hcqt_mag_{i}.png")

        plt.imshow(
            dphase[0, i, :, :],
            origin="lower",
            aspect="auto",
            cmap="inferno",
        )
        plt.savefig(f"figures/hcqt_dphase_{i}.png")


#---------------- Assigner function ----------------#

def pitch_activations_to_mf0_argmax(pitch_activation_mat, thresh):
    """
        Convert pitch activation map to pitch by argmaxing
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = np.argmax(pitch_activation_mat, axis=0)
    for i in range(peak_thresh_mat.shape[1]):
        peak_thresh_mat[peaks[i], i] = pitch_activation_mat[peaks[i], i]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = np.zeros(pitch_activation_mat.shape[1])

    for f, t in zip(idx[0], idx[1]):
        if f == 0:
            ## redundant because it has zeros already but making sure
            est_freqs[t] = 0
        else:
            if np.array(f).size > 1:
                idx_max = peak_thresh_mat[t, f].argmax()
                est_freqs[t] = freqs[f[idx_max]]
            else:
                est_freqs[t] = freqs[f]

    return times, est_freqs


def mf0_predict_batch(est_saliences, thresholds=[0.23, 0.17, 0.15, 0.17]):
    """
        Convert a batch of salience maps to multif0 by thresholding peak values
    """
    # construct the multi-pitch predictions
    predictions = np.zeros([est_saliences.shape[0], est_saliences.shape[3], 5]) # shape = (batch, time, 5)
    
    for i in range(est_saliences.shape[0]):
        timestamp, sop = pitch_activations_to_mf0_argmax(est_saliences[i,0], thresh=thresholds[0])
        _, alt = pitch_activations_to_mf0_argmax(est_saliences[i, 1], thresh=thresholds[1])
        _, ten = pitch_activations_to_mf0_argmax(est_saliences[i, 2], thresh=thresholds[2])
        _, bas = pitch_activations_to_mf0_argmax(est_saliences[i, 3], thresh=thresholds[3])
        
        predictions[i, :, 0] = timestamp
        predictions[i, :, 1] = sop       
        predictions[i, :, 2] = alt                
        predictions[i, :, 3] = ten
        predictions[i, :, 4] = bas

    return predictions


if __name__ == "__main__":
    test_hcqt()