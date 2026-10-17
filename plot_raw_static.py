"""
Raw Un-Normalized Range-Doppler Map Generator to visualize Static Reflections at 0 Hz.
"""

import os
import sys
import gc
import numpy as np
import scipy.signal as signal
import torch

from iq_reader import load_coherent_pair
from passive_radar_gpu import PassiveRadarGPU
import passive_radar as pr

file_a = "ch1_Fs95.1MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs95.1MHz_Fc640kHz_FMradio"

fc_raw = 95.1e6
fs_raw = 640000.0

cpi_sec = 4.0
target_bw = 200000.0
target_fs = 200000.0
max_range_km = 150.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load 4s slice
sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=int(cpi_sec * fs_raw), offset_samples=0)

# Shift to DC, LPF, Decimate
sub_len = min(len(sa_raw), 32768)
freqs, psd_a = signal.welch(sa_raw[:sub_len], fs=fs_raw, nperseg=2048, return_onesided=False)
peak_offset_hz = np.fft.fftshift(freqs)[np.argmax(np.fft.fftshift(psd_a))]
fc_effective = fc_raw + peak_offset_hz

t = np.arange(len(sa_raw), dtype=np.float32) / float(fs_raw)
shift_vec = np.exp(-1j * (2.0 * np.pi * peak_offset_hz) * t)

sa_tuned = sa_raw * shift_vec
sb_tuned = sb_raw * shift_vec
del sa_raw, sb_raw, shift_vec, t

cutoff_norm = (target_bw / 2.0) / (fs_raw / 2.0)
fir_coeff = signal.firwin(61, cutoff_norm, window='hamming')

sa_filt = signal.lfilter(fir_coeff, 1.0, sa_tuned)
sb_filt = signal.lfilter(fir_coeff, 1.0, sb_tuned)
del sa_tuned, sb_tuned

sa_dec = signal.resample_poly(sa_filt, 5, 16).astype(np.complex64)
sb_dec = signal.resample_poly(sb_filt, 5, 16).astype(np.complex64)
del sa_filt, sb_filt

sa_gpu = torch.tensor(sa_dec, dtype=torch.complex64, device=device)
sb_gpu = torch.tensor(sb_dec, dtype=torch.complex64, device=device)

pr_gpu = PassiveRadarGPU(device=device)

# RAW Surveillance signal (No ECA clutter filter)
s_sum_gpu = pr_gpu.beamform_sum_gpu(sa_gpu, sb_gpu, phase_offset_deg=0.0)
s_sur_raw_gpu = sb_gpu

# Compute RAW CAF Matrix
rdm_raw_db, r_km, d_hz, v_kmh = pr_gpu.compute_caf_gpu(
    s_sum_gpu, s_sur_raw_gpu, target_fs, fc_effective, max_range_km=max_range_km, max_doppler_hz=400.0
)

# Plot RAW Range-Doppler Map WITHOUT median subtraction
os.makedirs("output_plots", exist_ok=True)
raw_plot_path = "output_plots/ch1_ch2_95.1mhz_raw_static_reflections.png"

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    rdm_raw_db,
    aspect='auto',
    origin='lower',
    extent=[v_kmh[0], v_kmh[-1], r_km[0], r_km[-1]],
    cmap='viridis',
    vmin=-50.0,
    vmax=0.0
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Relative Power (dB)")
ax.set_title("RAW Un-Normalized Range-Doppler Map (Showing Static Reflections at 0 Hz Doppler)", fontsize=13, fontweight='bold')
ax.set_xlabel("Bistatic Velocity (km/h)")
ax.set_ylabel("Bistatic Range (km)")
ax.grid(True, linestyle="--", alpha=0.4, color="white")

plt.tight_layout()
plt.savefig(raw_plot_path, dpi=200)
plt.close()

print(f"Saved RAW Static Reflections Map -> {raw_plot_path}")
