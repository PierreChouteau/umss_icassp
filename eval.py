from faulthandler import disable
import os
from pathlib import Path
import pickle
import json
import argparse

import torch
import numpy as np
import pandas as pd
import librosa as lb

import data
import models
import utils
import evaluation_metrics as em
import ddsp.spectral_ops

from tqdm import tqdm
import time

import soundfile as sf
import mir_eval

torch.manual_seed(0)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--test-set', type=str, default='El Rossinyol', choices=['CSD', 'BCBQ', 'cantoria'])
parser.add_argument('--f0-from-mix', action='store_true', default=False)
parser.add_argument('--show-progress', action='store_true', default=False)
parser.add_argument('--compute', type=str, default='all', choices=['all', 'all_mask', 'all_no_mask', 'sp_SNR','sp_SI-SNR','mel_cep_dist','SI-SDR_mask','sp_SNR_mask','sp_SI-SNR_mask','mel_cep_dist_mask', 'f0_infos'])

args, _ = parser.parse_known_args()
tag = args.tag
is_u_net = tag[:4] == 'unet'

parser.add_argument('--eval-tag', type=str, default=tag)
args, _ = parser.parse_known_args()

f0_cuesta = args.f0_from_mix
print('f0-from-mix:', f0_cuesta, "\n")

if 'all' == args.compute:
    to_compute=['sp_SNR','sp_SI-SNR','mel_cep_dist','SI-SDR_mask','sp_SNR_mask','sp_SI-SNR_mask','mel_cep_dist_mask', 'f0_infos']
elif 'all_no_mask' == args.compute:
    to_compute=['sp_SNR','sp_SI-SNR','mel_cep_dist', 'f0_infos']
elif 'all_mask' == args.compute:
    to_compute=['SI-SDR_mask', 'sp_SNR_mask','sp_SI-SNR_mask','mel_cep_dist_mask', 'f0_infos']
else:
    to_compute = args.compute

# Identify what should be computed
compute_sp_snr = 'sp_SNR' in to_compute
compute_sp_si_snr = 'sp_SI-SNR' in to_compute
compute_mel_cep_dist = 'mel_cep_dist' in to_compute
compute_sp_snr_mask = 'sp_SNR_mask' in to_compute
compute_sp_si_snr_mask = 'sp_SI-SNR_mask' in to_compute
compute_si_sdr_mask = 'SI-SDR_mask' in to_compute
compute_mel_cep_dist_mask = 'mel_cep_dist_mask' in to_compute
compute_results = ('sp_SNR' in to_compute) or ('sp_SI-SNR' in to_compute) or ('mel_cep_dist' in to_compute) or ('f0_infos' in to_compute)
compute_results_masking = ('sp_SNR_mask' in to_compute) or ('sp_SI-SNR_mask' in to_compute) or ('SI-SDR_mask' in to_compute) or ('mel_cep_dist_mask' in to_compute) or ('f0_infos'in to_compute)

# Identify what should be computed for f0
compute_f0 = 'f0_infos' in to_compute 

# Load model arguments
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trained_model, model_args = utils.load_model(tag, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources = True
voices = model_args['voices'] if 'voices' in model_args.keys() else 'satb'
original_cunet = model_args['original_cu_net'] if 'original_cu_net' in model_args.keys() else False


mix_model = False
if mix_model:
    # Load d'un warmup monophonique
    name_model_mono = 'unsupervised_1s_string1song_crepe'
    model_path_1s = 'trained_models/{}'.format(name_model_mono)

    # load the best model in terms of validation loss
    target_model_path = next(Path(model_path_1s).glob("%s*.pth" % name_model_mono))
    state = torch.load(
        target_model_path,
        map_location=device
    )
    
    trained_model.load_state_dict(state, strict=False)


# Initialize results and results_masking path
if args.test_set == 'CSD': test_set_add_on = 'CSD'
elif args.test_set == 'BCBQ': test_set_add_on = 'BCBQ'
elif args.test_set == 'cantoria': test_set_add_on = 'Cantoria'
else: raise ValueError('Unknown test set')

# Initialize add_on element to path to save results
if args.f0_from_mix: f0_add_on = 'mf0'
else: f0_add_on = 'crepe'

# Initialize path to save results
path_to_save_results = 'evaluation/{}/eval_results_{}_{}_{}'.format(args.eval_tag, f0_add_on, test_set_add_on, device)
if not os.path.isdir(path_to_save_results):
    os.makedirs(path_to_save_results, exist_ok=True)

if is_u_net: path_to_save_results_masking = path_to_save_results
else:
    path_to_save_results_masking = 'evaluation/{}/eval_results_{}_{}_{}'.format(args.eval_tag + '_masking', f0_add_on, test_set_add_on, device)
    if not os.path.isdir(path_to_save_results_masking):
        os.makedirs(path_to_save_results_masking, exist_ok=True)


# Initialize test_set
if args.test_set == 'CSD':
    el_rossinyol = data.CSD(song_name='El Rossinyol', example_length=model_args['example_length'], allowed_voices=voices,
                        return_name=True, n_sources=model_args['n_sources'], singer_nb=[2], random_mixes=False,
                        f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    locus_iste = data.CSD(song_name='Locus Iste', example_length=model_args['example_length'], allowed_voices=voices,
                         return_name=True, n_sources=model_args['n_sources'], singer_nb=[3], random_mixes=False,
                         f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    nino_dios = data.CSD(song_name='Nino Dios', example_length=model_args['example_length'], allowed_voices=voices,
                     return_name=True, n_sources=model_args['n_sources'], singer_nb=[4], random_mixes=False,
                     f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    test_set = torch.utils.data.ConcatDataset([el_rossinyol, locus_iste, nino_dios])


elif args.test_set == 'BCBQ':
    bc_val = data.BCBQDataSets(data_set='BC',
                            validation_subset=True,
                            conf_threshold=0.4,
                            example_length=model_args['example_length'],
                            n_sources=model_args['n_sources'],
                            random_mixes=False,
                            return_name=True,
                            allowed_voices='satb',
                            f0_from_mix=f0_cuesta,
                            cunet_original=original_cunet,
                            cuesta_model=False,
                            cuesta_model_trainable=False)

    bq_val = data.BCBQDataSets(data_set='BQ',
                            validation_subset=True,
                            conf_threshold=0.4,
                            example_length=model_args['example_length'],
                            n_sources=model_args['n_sources'],
                            random_mixes=False,
                            return_name=True,
                            allowed_voices='satb',
                            f0_from_mix=f0_cuesta,
                            cunet_original=original_cunet,
                            cuesta_model=False,
                            cuesta_model_trainable=False)
    
    test_set = torch.utils.data.ConcatDataset([bc_val, bq_val])


elif args.test_set == 'cantoria':
    ejb1 = data.CantoriaDataSets(song_name='EJB1',
                                conf_threshold=0.4, 
                                example_length=model_args['example_length'], 
                                allowed_voices=voices,
                                return_name=True, 
                                n_sources=model_args['n_sources'], 
                                random_mixes=False, 
                                f0_from_mix=f0_cuesta, 
                                cunet_original=original_cunet, 
                                cuesta_model=False, 
                                cuesta_model_trainable=False,
                                )

    ejb2 = data.CantoriaDataSets(song_name='EJB2',
                                conf_threshold=0.4, 
                                example_length=model_args['example_length'], 
                                allowed_voices=voices,
                                return_name=True, 
                                n_sources=model_args['n_sources'], 
                                random_mixes=False, 
                                f0_from_mix=f0_cuesta, 
                                cunet_original=original_cunet, 
                                cuesta_model=False, 
                                cuesta_model_trainable=False,
                                )
    
    cea = data.CantoriaDataSets(song_name='CEA',
                                conf_threshold=0.4, 
                                example_length=model_args['example_length'], 
                                allowed_voices=voices,
                                return_name=True, 
                                n_sources=model_args['n_sources'], 
                                random_mixes=False, 
                                f0_from_mix=f0_cuesta, 
                                cunet_original=original_cunet, 
                                cuesta_model=False, 
                                cuesta_model_trainable=False,
                                )

    test_set = torch.utils.data.ConcatDataset([ejb1, ejb2, cea])

else: 
    raise ValueError('Unknown test set')


#Initialize both result dataframes
eval_results_dict={'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': []}
eval_results_masking_dict={'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': []}

if compute_sp_snr : eval_results_dict['sp_SNR']=[]
if compute_sp_si_snr : eval_results_dict['sp_SI-SNR']=[]
if compute_mel_cep_dist : eval_results_dict['mel_cep_dist']=[]
if compute_sp_snr_mask : eval_results_masking_dict['sp_SNR']=[]
if compute_sp_si_snr_mask : eval_results_masking_dict['sp_SI-SNR']=[]
if compute_si_sdr_mask : eval_results_masking_dict['SI-SDR']=[]
if compute_mel_cep_dist_mask : eval_results_masking_dict['mel_cep_dist']=[]

if compute_results:
    eval_results = pd.DataFrame(eval_results_dict)
if compute_results_masking:
    eval_results_masking = pd.DataFrame(eval_results_masking_dict)

pd.set_option("display.max_rows", None, "display.max_columns", None)

if is_u_net: n_seeds = 1
else: n_seeds = 1


# Load energy snippets
# index of signals to be removed in evaluation due to too low energy (considered as silence)
# mandatory to obtain the same results as in the paper
if model_args['n_sources'] == 4:
    if args.test_set == 'CSD': energy_snippet = pd.read_pickle('./Datasets/ChoralSingingDataset/energy_snippets_4s.pandas')
    elif args.test_set == 'cantoria': energy_snippet = pd.read_pickle('./Datasets/CantoriaDataset/energy_snippets_4s_bis.pandas')
elif model_args['n_sources'] == 2:
    energy_snippet = pd.read_pickle('./Datasets/ChoralSingingDataset/energy_snippets_2s.pandas')
    
energy_to_drop = []
for i, energy in enumerate(energy_snippet["energy"]):
    if energy < 10:
        for j in range(n_seeds):
            energy_to_drop.append(i + j * len(energy_snippet["energy"]))


# Evaluation loop
for seed in range(n_seeds):
    torch.manual_seed(seed)
    rng_state_torch = torch.get_rng_state()

    for idx in tqdm(range(len(test_set)), disable = not args.show_progress):
        # Load batch of 4 1-second frames
        d = test_set[idx]

        mix = d[0].to(device)
        f0_hz = d[1].to(device)
        # f0_hz = torch.zeros_like(d[1]).to(device)
        target_sources = d[2].to(device)
        name = d[3]
        voices = d[4]

        mix = mix.unsqueeze(0)
        target_sources = target_sources.unsqueeze(0)
        f0_hz = f0_hz[None, :, :]

        batch_size, n_samples, n_sources = target_sources.shape

        n_fft_metrics = 512
        overlap_metrics = 0.5

        # reset rng state so that each example gets the same state
        torch.random.set_rng_state(rng_state_torch)

        with torch.no_grad():

            if is_u_net:
                n_hop = int(trained_model.n_fft - trained_model.overlap * trained_model.n_fft)
                estimated_sources = utils.masking_unets_softmasks(trained_model, mix, f0_hz, n_sources,
                                                                  trained_model.n_fft, n_hop)
                # estimated_sources = utils.masking_unets2(trained_model, mix, f0_hz, n_sources,
                #                                          trained_model.n_fft, n_hop)

                source_estimates = torch.tensor(estimated_sources, device=device, dtype=torch.float32).unsqueeze(0)  # [batch_size, n_source, n_samples]
                source_estimates_masking = source_estimates.reshape((batch_size * n_sources, n_samples))
                source_export = source_estimates_masking.reshape((batch_size, n_sources, n_samples))

            else:
                if 'method' in model_args:
                    if model_args['method'] == 'reconstruction' or model_args['method'] == 'ste': mix_estimate, source_estimates, _, _, _, f0s = trained_model(mix, f0_hz)
                else: mix_estimate, source_estimates, _, _, f0s = trained_model(mix, f0_hz)
                                
                # [batch_size * n_sources, n_samples]
                source_estimates_masking = utils.masking_from_synth_signals_torch(mix, source_estimates, n_fft=2048, n_hop=256)
                source_export = source_estimates_masking.reshape((batch_size, n_sources, n_samples))

            target_sources = target_sources.transpose(1, 2)  # [batch_size, n_sources, n_samples]
            target_sources = target_sources.reshape((batch_size * n_sources, n_samples))
            source_estimates = source_estimates.reshape((batch_size * n_sources, n_samples))

            # Export source estimates as wav files with sounfiles
            export_sources = True
            if export_sources:
                for i in range(batch_size):
                    for j in range(n_sources):
                        if not os.path.isdir(path_to_save_results_masking + '/sources_estimates_masking/'): os.makedirs(path_to_save_results_masking + '/sources_estimates_masking/', exist_ok=True)
                        sf.write(path_to_save_results_masking + '/sources_estimates_masking' + f'/sources_estimates_masking_{name}_voice_{voices[j]}.wav', source_export[i, j].cpu().numpy(), 16000)
                            
                        if not os.path.isdir(path_to_save_results_masking + '/sources_estimates'): os.makedirs(path_to_save_results_masking + '/sources_estimates/', exist_ok=True)
                        sf.write(path_to_save_results_masking + '/sources_estimates' + f'/sources_estimates_{name}_voice_{voices[j]}.wav', source_estimates[j].cpu().numpy(), 16000)
                        
                        if not os.path.isdir(path_to_save_results_masking + '/target_sources'): os.makedirs(path_to_save_results_masking + '/target_sources/', exist_ok=True)
                        sf.write(path_to_save_results_masking + '/target_sources' + f'/target_sources_{name}_voice_{voices[j]}.wav', target_sources[j].cpu().numpy(), 16000)
                    
                    if not os.path.isdir(path_to_save_results_masking + '/mix'): os.makedirs(path_to_save_results_masking + '/mix/', exist_ok=True)
                    sf.write(path_to_save_results_masking + '/mix' + f'/mix_{name}.wav', mix[i].cpu().numpy(), 16000)
                    
                    if not os.path.isdir(path_to_save_results_masking + '/mix_reconstruct') and not is_u_net: os.makedirs(path_to_save_results_masking + '/mix_reconstruct/', exist_ok=True)
                    if not is_u_net: sf.write(path_to_save_results_masking + '/mix_reconstruct' + f'/mix_reconstruct_{name}.wav', mix_estimate[i].cpu().numpy(), 16000)
            
            # compute metrics for f0s
            if compute_f0 and not is_u_net:

                n_eval_frames = int(np.ceil(n_samples / 16000))
                
                overall_accuracies = np.zeros((batch_size, n_sources, n_eval_frames))
                raw_pitches_accuracies = np.zeros((batch_size, n_sources, n_eval_frames))
                raw_chroma_accuracies = np.zeros((batch_size, n_sources, n_eval_frames))
                voicing_false_alarm_rates = np.zeros((batch_size, n_sources, n_eval_frames))
                voicing_recall_rates = np.zeros((batch_size, n_sources, n_eval_frames))
                f_scores = np.zeros((batch_size, n_sources, n_eval_frames))
                
                for source in range(n_sources):
                    for n in range(n_eval_frames):
                        ref_freq = f0_hz[0, n:(n+1)*int(250/4), source].cpu().numpy()
                        ref_time = np.arange(len(ref_freq)) * 0.016
                        
                        est_freq = f0s[source, n:(n+1)*int(250/4), 0].cpu().numpy()
                        (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time, ref_freq,
                                                                                       ref_time, est_freq)
                        
                        overall_accuracies[0, source, n] = mir_eval.melody.overall_accuracy(ref_v, ref_c,
                                                                                            est_v, est_c)

                        raw_pitches_accuracies[0, source, n] = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,
                                                                                                  est_v, est_c)
                        
                        raw_chroma_accuracies[0, source, n] = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
                        
                        voicing_recall_rates[0, source, n] = mir_eval.melody.voicing_recall(ref_v, est_v)
                        
                        voicing_false_alarm_rates[0, source, n] = mir_eval.melody.voicing_false_alarm(ref_v, est_v)
                        
                        metrics = mir_eval.multipitch.metrics(ref_time, ref_freq[:, None]+20, ref_time, est_freq[:, None]+20, min_freq=0)
                        precision, recall, accuracy = metrics[0:3]
                        
                        f_scores[0, source, n] = 2 * precision * recall / (precision + recall + 1e-8)
                        
                                    
            # compute spectral SNR masking
            if compute_sp_si_snr_mask :
                si_snr_masking = em.spectral_si_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
                si_snr_masking = si_snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SNR masking
            if compute_sp_snr_mask :
                snr_masking = em.spectral_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
                snr_masking = snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute mel cepstral distance masking
            if compute_mel_cep_dist_mask :
                mcd_masking = em.mel_cepstral_distance(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
                mcd_masking = mcd_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute SI-SDR masking
            if compute_si_sdr_mask :
                si_sdr_masking = em.si_sdr(target_sources, source_estimates_masking)
                si_sdr_masking = si_sdr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()
                
            # compute spectral SNR
            if compute_sp_snr : 
                snr = em.spectral_snr(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                snr = snr.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SI-SNR
            if compute_sp_si_snr :
                si_snr = em.spectral_si_snr(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                si_snr = si_snr.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute mel cepstral distance
            if compute_mel_cep_dist :
                mcd = em.mel_cepstral_distance(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
                mcd = mcd.reshape((batch_size, n_sources, -1)).cpu().numpy()

        # n_eval_frames = snr.shape[-1]
        n_eval_frames = 4

        mix_names = [name for _ in range(n_sources * n_eval_frames)]
        voice = [v for v in voices for _ in range(n_eval_frames)]
        eval_frame = [f for _ in range(n_sources * batch_size) for f in range(n_eval_frames)]
        seed_results = [seed] * len(eval_frame)

        # append results of each iteration into eval_results_masking 
        if compute_results_masking:
            batch_results_masking_dict={'mix_name': mix_names, 'eval_seed': seed_results, 'voice': voice, 'eval_frame': eval_frame}
            
            if compute_sp_snr_mask :
                snr_results_masking = [snr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['sp_SNR'] = snr_results_masking
            if compute_sp_si_snr_mask :
                si_snr_results_masking = [si_snr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['sp_SI-SNR'] = si_snr_results_masking
            if compute_si_sdr_mask :
                si_sdr_results_masking = [si_sdr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['SI-SDR'] = si_sdr_results_masking
            if compute_mel_cep_dist_mask :
                mcd_results_masking = [mcd_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['mel_cep_dist'] = mcd_results_masking
            
            if compute_f0 and not is_u_net:
                f0_overall_accuracy_results = [overall_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['Overall-Accuracy'] = f0_overall_accuracy_results
                
                f0_raw_pitches_accuracies_results = [raw_pitches_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['Raw-Pitch-Accuracy'] = f0_raw_pitches_accuracies_results
                
                f0_raw_chroma_accuracies_results = [raw_chroma_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['Raw-Chroma-Accuracy'] = f0_raw_chroma_accuracies_results
                
                f_score_results = [f_scores[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['F-Score'] = f_score_results
            
                voicing_false_alarm_results = [voicing_false_alarm_rates[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['Voicing-False-Alarm'] = voicing_false_alarm_results
            
                voicing_recall_results = [voicing_recall_rates[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_masking_dict['Voicing-Recall'] = voicing_recall_results            
            
            batch_results_masking = pd.DataFrame(batch_results_masking_dict)
            eval_results_masking = eval_results_masking.append(batch_results_masking, ignore_index=True)

        # append results of each iteration into eval_results
        if compute_results:
            batch_results_dict={'mix_name': mix_names, 'eval_seed': seed_results, 'voice': voice, 'eval_frame': eval_frame}

            if compute_sp_snr : 
                snr_results = [snr[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['sp_SNR'] = snr_results
            if compute_sp_si_snr :
                si_snr_results = [si_snr[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['sp_SI-SNR'] = si_snr_results
            if compute_mel_cep_dist :
                mcd_results = [mcd[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['mel_cep_dist'] = mcd_results
                
            if compute_f0 and not is_u_net:
                f0_overall_accuracy_results = [overall_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['Overall-Accuracy'] = f0_overall_accuracy_results
                
                f0_raw_pitches_accuracies_results = [raw_pitches_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['Raw-Pitch-Accuracy'] = f0_raw_pitches_accuracies_results
                
                f0_raw_chroma_accuracies_results = [raw_chroma_accuracies[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['Raw-Chroma-Accuracy'] = f0_raw_chroma_accuracies_results
                
                f_scores_restults = [f_scores[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['F-Score'] = f_scores_restults

                voicing_false_alarm_results = [voicing_false_alarm_rates[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['Voicing-False-Alarm'] = voicing_false_alarm_results
            
                voicing_recall_results = [voicing_recall_rates[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
                batch_results_dict['Voicing-Recall'] = voicing_recall_results      

            batch_results = pd.DataFrame(batch_results_dict)
            eval_results = eval_results.append(batch_results, ignore_index=True)



# Export and print results
if compute_results:
    if not is_u_net: 
        # drop results for energy below threshold
        if args.test_set == 'CSD': eval_results = eval_results.drop(energy_to_drop)
        elif args.test_set == 'cantoria': eval_results = eval_results.drop(energy_to_drop)
        
        # save data frame with all results
        eval_results.to_pickle(path_to_save_results + '/all_results.pandas')

    # compute mean, median, std over all voices and mixes and eval_frames
    means = eval_results.mean(axis=0, skipna=True, numeric_only=True)
    medians = eval_results.median(axis=0, skipna=True, numeric_only=True)
    stds = eval_results.std(axis=0, skipna=True, numeric_only=True)

    print("\n" + "-----------------------" + "\n" + tag + "\n")
    if compute_sp_snr:
        print('sp_SNR:', 'mean', means['sp_SNR'], 'median', medians['sp_SNR'], 'std', stds['sp_SNR'])
    if compute_sp_si_snr:
        print('sp_SI-SNR', 'mean', means['sp_SI-SNR'], 'median', medians['sp_SI-SNR'], 'std', stds['sp_SI-SNR'])
    if compute_mel_cep_dist:
        print('mel cepstral distance', 'mean', means['mel_cep_dist'], 'median', medians['mel_cep_dist'], 'std', stds['mel_cep_dist'])
    if compute_f0 and not is_u_net:
        print('\n'+'Voicing Recall:', 'mean', means['Voicing-Recall'], 'median', medians['Voicing-Recall'], 'std', stds['Voicing-Recall'])
        print('Voicing False Alarm:', 'mean', means['Voicing-False-Alarm'], 'median', medians['Voicing-False-Alarm'], 'std', stds['Voicing-False-Alarm'])
        print('f0 Raw Pitch Accuracy:', 'mean', means['Raw-Pitch-Accuracy'], 'median', medians['Raw-Pitch-Accuracy'], 'std', stds['Raw-Pitch-Accuracy'])
        print('f0 Overall Accuracy:', 'mean', means['Overall-Accuracy'], 'median', medians['Overall-Accuracy'], 'std', stds['Overall-Accuracy'])
        print('f0 Raw Chroma Accuracy:', 'mean', means['Raw-Chroma-Accuracy'], 'median', medians['Raw-Chroma-Accuracy'], 'std', stds['Raw-Chroma-Accuracy'])
        print('f0 F-Score:', 'mean', means['F-Score'], 'median', medians['F-Score'], 'std', stds['F-Score'])
    print("-----------------------" + "\n")


if compute_results_masking:
    
    # drop results for energy below threshold
    if args.test_set == 'CSD': eval_results_masking = eval_results_masking.drop(energy_to_drop)
    elif args.test_set == 'cantoria': eval_results_masking = eval_results_masking.drop(energy_to_drop)
    
    # save data frame with all results
    eval_results_masking.to_pickle(path_to_save_results_masking + '/all_results.pandas')

    # compute mean, median, std over all voices and mixes and eval_frames
    means_masking = eval_results_masking.mean(axis=0, skipna=True, numeric_only=True)
    medians_masking = eval_results_masking.median(axis=0, skipna=True, numeric_only=True)
    stds_masking = eval_results_masking.std(axis=0, skipna=True, numeric_only=True)

    print("\n" + "-----------------------" + "\n" + tag + '_masking' + "\n")
    if compute_si_sdr_mask:
        print('SI-SDR', 'mean', means_masking['SI-SDR'], 'median', medians_masking['SI-SDR'], 'std', stds_masking['SI-SDR'])
    if compute_sp_snr_mask:
        print('sp_SNR', 'mean', means_masking['sp_SNR'], 'median', medians_masking['sp_SNR'], 'std', stds_masking['sp_SNR'])
    if compute_sp_si_snr_mask:    
        print('sp_SI-SNR', 'mean', means_masking['sp_SI-SNR'], 'median', medians_masking['sp_SI-SNR'], 'std', stds_masking['sp_SI-SNR'])
    if compute_mel_cep_dist_mask:    
        print('mel cepstral distance:', 'mean', means_masking['mel_cep_dist'], 'median', medians_masking['mel_cep_dist'], 'std', stds_masking['mel_cep_dist'])
    if compute_f0 and not is_u_net:
        print('\n'+'Voicing Recall:', 'mean', means['Voicing-Recall'], 'median', medians['Voicing-Recall'], 'std', stds['Voicing-Recall'])
        print('Voicing False Alarm:', 'mean', means['Voicing-False-Alarm'], 'median', medians['Voicing-False-Alarm'], 'std', stds['Voicing-False-Alarm'])
        print('f0 Raw Pitch Accuracy:', 'mean', means['Raw-Pitch-Accuracy'], 'median', medians['Raw-Pitch-Accuracy'], 'std', stds['Raw-Pitch-Accuracy'])
        print('f0 Overall Accuracy:', 'mean', means['Overall-Accuracy'], 'median', medians['Overall-Accuracy'], 'std', stds['Overall-Accuracy'])
        print('f0 Raw Chroma Accuracy:', 'mean', means['Raw-Chroma-Accuracy'], 'median', medians['Raw-Chroma-Accuracy'], 'std', stds['Raw-Chroma-Accuracy'])
        print('f0 F-Score:', 'mean', means['F-Score'], 'median', medians['F-Score'], 'std', stds['F-Score'])
    print("-----------------------" + "\n")