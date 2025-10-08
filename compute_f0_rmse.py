# %%
import os
import librosa
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import pysptk
from fastdtw import fastdtw
import soundfile as sf
from scipy import spatial

# %%
gt_path = './testset_wavs/wavs_gt_valset'
gt_hifigan_path = './testset_wavs/wavs_hifigan_valset'
matchatts_d_vecor_path = './testset_wavs/wavs_matchatts_valset'
prosodyfm_path = './testset_wavs/wavs_prosodyfm_valset'
with_only_b_path = './testset_wavs/wavs_with_only_b_valset'
with_only_gst_path = './testset_wavs/wavs_with_only_gst_valset'


# %%
def _get_best_mcep_params(fs: int):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")

# %%
def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """Extract World-based acoustic features.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0

# %%
def ave_f0_rmse(gt_path, syn_path, fs = 22050, n_fft = 1024, n_shift = 256, mcep_dim = 34, mcep_alpha = 0.45, f0min = 40, f0max = 800):
    f0_rmse_list = []
    
    for file_name in os.listdir(syn_path):
        gt_file = os.path.join(gt_path, file_name)
        syn_file = os.path.join(syn_path, file_name)

        gen_x, _ = sf.read(syn_file, dtype="int16")
        gt_x, _ = sf.read(gt_file, dtype="int16")

        # extract ground truth and converted features
        gen_mcep, gen_f0 = world_extract(
                x=gen_x,
                fs=fs,
                f0min=f0min,
                f0max=f0max,
                n_fft=n_fft,
                n_shift=n_shift,
                mcep_dim=mcep_dim,
                mcep_alpha=mcep_alpha,
            )
        gt_mcep, gt_f0 = world_extract(
                x=gt_x,
                fs=fs,
                f0min=f0min,
                f0max=f0max,
                n_fft=n_fft,
                n_shift=n_shift,
                mcep_dim=mcep_dim,
                mcep_alpha=mcep_alpha,
            )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_f0_dtw = gen_f0[twf[0]]
        gt_f0_dtw = gt_f0[twf[1]]

        # Get voiced part
        nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
        gen_f0_dtw_voiced = np.log(gen_f0_dtw[nonzero_idxs])
        gt_f0_dtw_voiced = np.log(gt_f0_dtw[nonzero_idxs])

        # log F0 RMSE
        if gen_f0_dtw_voiced.size == 0 or gt_f0_dtw_voiced.size == 0:
            print(f"Skipping {file_name} due to empty voiced frames.")
            continue
        log_f0_rmse = np.sqrt(np.mean((gen_f0_dtw_voiced - gt_f0_dtw_voiced) ** 2))
        f0_rmse_list.append(log_f0_rmse)

    avg_f0_rmse = np.mean(f0_rmse_list)         
    return avg_f0_rmse
        

# %%
#print('styletts2', ave_f0_rmse(gt_path, styletts2_path))
#print('prosodyfm', ave_f0_rmse(gt_path, prosodyfm_path))
#print('stylespeech', ave_f0_rmse(gt_path, stylespeech_path))

#print('matchatts_trained_with_vctk', ave_f0_rmse(gt_path, matchatts_trained_with_vctk))
#print('prosodyfm_trained_with_vctk', ave_f0_rmse(gt_path, prosodyfm_trained_with_vctk))
#print('gt_hifigan_valset', ave_f0_rmse(gt_path, gt_hifigan_path))
print('matchatts_valset', ave_f0_rmse(gt_path, matchatts_d_vecor_path))
#print('prosodyfm_valset', ave_f0_rmse(gt_path, prosodyfm_path))
#print('with_only_b_valset', ave_f0_rmse(gt_path, with_only_b_path))
#print('with_only_gst_valset', ave_f0_rmse(gt_path, with_only_gst_path))
