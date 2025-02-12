"""
Inference script to perform audio source separation using a trained model.
"""

import os
import argparse

import torch
import torchaudio
import soundfile as sf

import utils


def load_audio(audio_path, sample_rate):
  """
  Load an audio file, convert it to mono, and resample it to a specified
  sample rate.

  Parameters
  ----------
  audio_path : str
    Path to the audio file.
  sample_rate : int
    The desired sample rate to resample the audio to.

  Returns
  -------
  audio: torch.Tensor
    The loaded and resampled audio as a tensor.
  """

  # load the audio file and convert it to mono
  audio, sr = torchaudio.load(audio_path)
  audio = audio.mean(dim=0, keepdim=True)

  # resample the audio to 16kHz if necessary
  if sr != sample_rate:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=sample_rate)
    audio = resampler(audio)

  return audio


def slice_audio(audio, sample_rate, audio_length, step_size):
  """
  Slices the input audio into segments of specified length with a given step
  size.

  Parameters
  ----------
  audio : torch.Tensor
    The input audio tensor of shape (1, samples).
  sample_rate : int
    The sample rate of the audio.
  audio_length : float
    The length of each audio segment in seconds.
  step_size : float
    The step size in seconds for moving the window to create segments.

  Returns
  -------
  audio_segments : torch.Tensor
    A tensor containing the sliced audio segments of shape
    (n_segments, segment_size).
  """

  _, n_samples = audio.shape
  segment_size = int(audio_length * sample_rate)
  step_size = int(step_size * sample_rate)

  n_segments = int(1 + (n_samples - (segment_size + 1)) // step_size)
  audio_segments = torch.zeros((n_segments + 1, segment_size))

  for i in range(n_segments):
    start = i * step_size
    end = start + segment_size
    audio_segments[i] = audio[:, start:end]

  if n_segments == 0:
    len_last = audio.shape[-1]
    audio_segments[0][:len_last] = audio
  else:
    len_last = audio[:, end:].shape[-1]
    audio_segments[i + 1][:len_last] = audio[:, end:]

  return audio_segments


def save_audio_segments(
        output_dir,
        audio_segments,
        estimated_sources,
        estimated_mix,
        original_audio_size,
        sample_rate,
        mode="segmented_audio",
):
  """
  Save audio segments to WAV files.

  Parameters
  ----------
  output_dir : str
    Directory to save the separated audio files.
  audio_segments : torch.Tensor
    Tensor containing the original audio segments.
  estimated_sources : torch.Tensor
    Tensor containing the source estimates for each segment.
  estimated_mix : torch.Tensor
    Tensor containing the reconstructed mix estimates for each segment.
  n_segments : int
    Number of audio segments.
  n_sources : int
    Number of sources in each segment.
  sample_rate : int
    Sample rate for the audio files.
  mode : str
    Mode to save the audio files.
    - If "full_audio", the audio files are saved as full audio files.
    - If "segmented_audio", the audio files are saved as segmented
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  n_segments = estimated_sources.shape[0]
  n_sources = estimated_sources.shape[1]
  segment_size = estimated_sources.shape[2]

  if mode == "full_audio":

    # concat all the segments to get the final source estimates and mix
    estimated_sources = estimated_sources.transpose(0, 1)  # shape (n_sources, n_segments, segment_size) # nopep8
    estimated_sources = estimated_sources.reshape(n_sources, n_segments * segment_size)  # shape (n_sources, n_segments * segment_size) # nopep8

    estimated_mix = estimated_mix.reshape(n_segments * segment_size)  # shape (n_segments * segment_size) # nopep8
    audio_segments = audio_segments.reshape(n_segments * segment_size)  # shape (n_segments * segment_size) # nopep8

    # crop the audio to the original size
    estimated_sources = estimated_sources[:, :original_audio_size]
    estimated_mix = estimated_mix[:original_audio_size]

    for j in range(n_sources):
      sf.write(
          file=f"{output_dir}/estimated_source_voice_{j}.wav",
          data=estimated_sources[j].cpu().numpy(),
          samplerate=sample_rate,
      )
      sf.write(
          file=f"{output_dir}/mix.wav",
          data=audio_segments.cpu().numpy(),
          samplerate=sample_rate,
      )
      sf.write(
          file=f"{output_dir}/mix_reconstruct.wav",
          data=estimated_mix.cpu().numpy(),
          samplerate=sample_rate,
      )

  elif mode == "segmented_audio":
    for i in range(n_segments):
      for j in range(n_sources):
        sf.write(
            file=f"{output_dir}/estimated_source_segment_{i}_voice_{j}.wav",
            data=estimated_sources[i, j].cpu().numpy(),
            samplerate=sample_rate,
        )
      sf.write(
          file=f"{output_dir}/mix_segment_{i}.wav",
          data=audio_segments[i].cpu().numpy(),
          samplerate=sample_rate,
      )
      sf.write(
          file=f"{output_dir}/mix_segment_reconstruct_{i}.wav",
          data=estimated_mix[i].cpu().numpy(),
          samplerate=sample_rate,
      )


def main(args):
  """
  Main function to perform audio source separation using a trained model.

  Parameters
  ----------
  args : dict
    Dictionary containing the following keys:
    - "audio_path" (str): Path to the audio file.
    - "tag" (str): Tag to identify the trained model.
    - "mode" (str): Mode to save the audio files.
    - "output_dir" (str): Directory to save the separated audio files.
    - "device" (str): Device to run the model on (e.g., "cpu" or "cuda").
  """

  # load the trained model
  trained_model = utils.load_model(
      tag=args["tag"],
      device=args["device"],
      return_args=False,
  )
  trained_model.return_synth_params = False
  trained_model.return_sources = True

  # Parameters for audio processing, these should match the training parameters
  sample_rate = 16000
  audio_length = 4
  step_size = 4
  n_sources = 4
  n_fft = 2048
  n_hop = 256

  # Load the audio file and slice it into segments
  audio = load_audio(args["audio_path"], sample_rate)
  audio_segments = slice_audio(audio, sample_rate, audio_length, step_size)  # shape (n_segments, segment_size) # nopep8
  audio_segments = audio_segments.to(device=args["device"])

  with torch.no_grad():

    n_segments = audio_segments.shape[0]
    segment_size = audio_segments.shape[1]

    # forward pass through the model
    estimated_mix, estimated_sources, _, _, _, _ = trained_model(
        audio_segments, None)

    # get the source estimates
    estimated_sources_masking = utils.masking_from_synth_signals_torch(
        true_mix=audio_segments,
        estimated_sources=estimated_sources,
        n_fft=n_fft,
        n_hop=n_hop,
    )

    # reshape the source estimates to (n_segments, n_sources, segment_size)
    estimated_sources_masking = estimated_sources_masking.reshape(
        (n_segments, n_sources, segment_size))

    # save the source estimates and the mixtures
    save_audio_segments(
        output_dir=args["output_dir"],
        audio_segments=audio_segments,
        estimated_sources=estimated_sources_masking,
        estimated_mix=estimated_mix,
        original_audio_size=audio.shape[-1],
        sample_rate=sample_rate,
        mode=args["mode"]
    )


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--audio_path",
      type=str,
      required=True,
      help="Path to an audio file.",
  )
  parser.add_argument(
      "--tag",
      type=str,
      choices=["W-Up_bcbq", "W-Up_bc1song",
               "Sft-Sft_bcbq", "Sft-Sft_bc1song",
               "Sf-Sft_bcbq", "Sf-Sft_bc1song",
               "Sf-Sf_bcbq", "Sf-Sf_bc1song"],
      default="W-Up_bcbq",
      help="Tag to identify the trained model to use.",
  )
  parser.add_argument(
      "--mode",
      type=str,
      choices=["segmented_audio", "full_audio"],
      default="segmented_audio",
      help="Mode to save the audio files.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./output",
      help="Directory to save the separated audio files.",
  )
  parser.add_argument(
      "--device",
      type=str,
      choices=["cpu", "cuda"],
      default="cpu",
      help="Device to run the model on (e.g., 'cpu' or 'cuda').",
  )

  args = parser.parse_args()
  args = vars(args)

  main(args)
