# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py

import librosa
from librosa.util.utils import fix_length

import scipy
import numpy as np 

import torch

import torchaudio
from nnAudio import features

from models.ddsp.core import safe_log, pad_for_stft
from models.ddsp.core import torch_float32


def stft(audio, frame_size=2048, overlap=0.75, center=False, pad_end=True):
    """Differentiable stft in PyTorch, computed in batch."""
    audio = torch_float32(audio)
    hop_length = int(frame_size * (1.0 - overlap))
    if pad_end:
        # pad signal so that STFT window is slid until it is
        # completely beyond the signal
        audio = pad_for_stft(audio, frame_size, hop_length)
    assert frame_size * overlap % 2.0 == 0.0
    window = torch.hann_window(int(frame_size), device=audio.device)
    s = torch.stft(
        input=audio,
        n_fft=int(frame_size),
        hop_length=hop_length,
        win_length=int(frame_size),
        window=window,
        center=center)
    return s

def istft(stft, frame_size=2048, overlap=0.75, center=False, length=64000):
    """Differentiable istft in PyTorch, computed in batch."""

    # stft [batch_size, fft_size//2 + 1, n_frames, 2]

    stft = torch_float32(stft)
    hop_length = int(frame_size * (1.0 - overlap))

    assert frame_size * overlap % 2.0 == 0.0
    window = torch.hann_window(int(frame_size), device=stft.device)
    s = torch.istft(
        input=stft,
        n_fft=int(frame_size),
        hop_length=hop_length,
        win_length=int(frame_size),
        window=window,
        center=center,
        length=length)
    return s



def compute_mag(audio, size=2048, overlap=0.75, pad_end=True, center=False, add_in_sqrt=0.0):
    stft_cmplx = stft(audio, frame_size=size, overlap=overlap, center=center, pad_end=pad_end)
    # add_in_sqrt is added before sqrt is taken because the gradient of torch.sqrt(0) is NaN
    mag = torch.sqrt(stft_cmplx[..., 0]**2 + stft_cmplx[..., 1]**2 + add_in_sqrt)
    return torch_float32(mag)

def compute_mel(audio,
                sr=16000,
                lo_hz=20.0,
                hi_hz=8000.0,
                bins=229,
                fft_size=2048,
                overlap=0.75,
                pad_end=True,
                add_in_sqrt=0.0):

    mag = compute_mag(audio, fft_size, overlap, pad_end, center=False, add_in_sqrt=add_in_sqrt)

    mel = torchaudio.transforms.MelScale(n_mels=bins,
                                         sample_rate=sr,
                                         f_min=lo_hz,
                                         f_max=hi_hz).to(mag.device)(mag)
    return mel

def compute_logmel(audio,
                   sr=16000,
                   lo_hz=20.0,
                   hi_hz=8000.0,
                   bins=229,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   add_in_sqrt=0.0):

    mel = compute_mel(audio, sr, lo_hz, hi_hz, bins, fft_size, overlap, pad_end, add_in_sqrt)
    return safe_log(mel)

def compute_logmag(audio,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True):
    mag = compute_mag(audio, fft_size, overlap, pad_end)
    return safe_log(mag)


def diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis.
    Args:
      x: Input tensor of any dimension.
      axis: Axis on which to take the finite difference.
    Returns:
      d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
      ValueError: Axis out of range for tensor.
    """
    shape = list(x.shape)
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                         (axis, len(shape)))

    begin_back = [0 for _ in range(len(shape))]
    begin_front = [0 for _ in range(len(shape))]
    begin_front[axis] = 1

    shape[axis] -= 1
    slice_front = slice_tf(x, begin_front, shape)
    slice_back = slice_tf(x, begin_back, shape)
    d = slice_front - slice_back
    return d


def slice_tf(input, begin, size):
    """mimic tf.slice

    This operation extracts a slice of size size from a tensor input
    starting at the location specified by begin. The slice size is
    represented as a tensor shape, where size[i] is the number of
    elements of the 'i'th dimension of input_ that you want to slice.
    The starting location (begin) for the slice is represented as an
    offset in each dimension of input. In other words, begin[i] is the
    offset into the i'th dimension of input that you want to slice from.
    """
    dims = len(input.shape)
    if dims == 1:
        slice = input[begin[0]:begin[0]+size[0]]
    elif dims == 2:
        slice = input[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1]]
    elif dims == 3:
        slice = input[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1], begin[2]:begin[2]+size[2]]
    elif dims == 4:
        slice = input[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1],
                      begin[2]:begin[2]+size[2], begin[3]:begin[3]+size[3]]
    else:
        raise NotImplementedError("slice does not support more than 4 dimensions at the moment")
    return slice



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
    """Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _, over_sample) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        n_octaves * 12 * over_sample, f_min, bins_per_octave=bins_per_octave)
    return freq_grid


def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length, _) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def hcqt_torch(audio, device):
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
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        mag = librosa.amplitude_to_db(mag.cpu(), ref=np.max)
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


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values
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
    return times, est_freqs