"""
Phased Array Beamforming & Spatial Nulling Processor for Coherent Dual-Channel Passive Radar.

Calculates:
1. Cross-correlation & exact phase offset delta_phi between Channel 1 and Channel 2.
2. Max Channel (S_max): Coherently sums Ch1 and Ch2 to maximize direct FM signal.
3. Null Channel (S_null): Subtraher Ch1 and Ch2 to form deep spatial null towards FM transmitter.
4. Range-Doppler maps & radar correlations for different signal combinations.
"""

import os
import sys
import gc
import time
import numpy as np
import scipy.signal as signal
import torch

from iq_reader import load_coherent_pair
from passive_radar_gpu import PassiveRadarGPU
import passive_radar as pr

file_a = "ch1_Fs102MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs102MHz_Fc640kHz_FMradio"

fc_raw = 102.0e6
fs_raw = 640000.0

cpi_sec = 4.0
target_bw = 200000.0
target_fs = 200000.0
max_range_km = 100.0

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=======================================================================")
print("PHASED ARRAY DIRECT SIGNAL MAXIMIZATION & SPATIAL NULLING RADAR")
print("=======================================================================")
print(f"  Files: {file_a} & {file_b}")
print(f"  Fc: {fc_raw/1e6:.1f} MHz | Fs: {fs_raw/1e3:.0f} kHz | CPI: {cpi_sec:.1f} s")
print("=======================================================================\n")

# 1. Load IQ data
num_raw = int(cpi_sec * fs_raw)
sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw, offset_samples=0)

# 2. Detect FM carrier peak
sub_len = min(len(sa_raw), 32768)
freqs, psd_a = signal.welch(sa_raw[:sub_len], fs=fs_raw, nperseg=2048, return_onesided=False)
peak_offset_hz = np.fft.fftshift(freqs)[np.argmax(np.fft.fftshift(psd_a))]
fc_effective = fc_raw + peak_offset_hz

print(f"1. Detected FM Carrier Offset: {peak_offset_hz/1e3:+.2f} kHz (Fc = {fc_effective/1e6:.4f} MHz)")

# Shift to DC, LPF, Decimate to 200 kHz
t = np.arange(len(sa_raw), dtype=np.float32) / float(fs_raw)
shift_vec = np.exp(-1j * (2.0 * np.pi * peak_offset_hz) * t)

sa_tuned = sa_raw * shift_vec
sb_tuned = sb_raw * shift_vec
del sa_raw, sb_raw, shift_vec, t
gc.collect()

cutoff_norm = (target_bw / 2.0) / (fs_raw / 2.0)
fir_coeff = signal.firwin(61, cutoff_norm, window='hamming')

sa_filt = signal.lfilter(fir_coeff, 1.0, sa_tuned)
sb_filt = signal.lfilter(fir_coeff, 1.0, sb_tuned)
del sa_tuned, sb_tuned
gc.collect()

sa_dec = signal.resample_poly(sa_filt, 5, 16).astype(np.complex64)
sb_dec = signal.resample_poly(sb_filt, 5, 16).astype(np.complex64)
del sa_filt, sb_filt
gc.collect()

# 3. Calculate Cross-Correlation between Channel 1 and Channel 2
print("\n2. Calculating Cross-Correlation & Direct Path Phase Difference...")
cross_corr = np.sum(sa_dec * np.conj(sb_dec))
delta_phi_rad = np.angle(cross_corr)
delta_phi_deg = np.degrees(delta_phi_rad)
coherence = np.abs(cross_corr) / (np.sqrt(np.sum(np.abs(sa_dec)**2)) * np.sqrt(np.sum(np.abs(sb_dec)**2)))

print(f"   Cross-Channel Complex Correlation Angle (Direct Path Phase Offset): {delta_phi_deg:+.2f}° ({delta_phi_rad:+.4f} rad)")
print(f"   Cross-Channel Coherence Magnitude: {coherence:.4f}")

# Measure individual channel powers
p_ch1 = np.mean(np.abs(sa_dec)**2)
p_ch2 = np.mean(np.abs(sb_dec)**2)

# 4. Construct MAX Channel (Case a)
# S_max = S1 + S2 * exp(-j * delta_phi)
s_max = sa_dec + sb_dec * np.exp(-1j * delta_phi_rad)
p_max = np.mean(np.abs(s_max)**2)

# 5. Construct NULL Channel (Case b)
# S_null = S1 - S2 * exp(-j * delta_phi) = S1 + S2 * exp(j * (pi - delta_phi))
s_null = sa_dec - sb_dec * np.exp(-1j * delta_phi_rad)
p_null = np.mean(np.abs(s_null)**2)

direct_gain_db = 10.0 * np.log10(p_max / max(p_ch1, p_ch2))
null_depth_db = 10.0 * np.log10(p_null / p_max)

print("\n3. Power Analysis & Spatial Beamforming Performance:")
print(f"   Channel 1 Power:  {p_ch1:.3e} ({10*np.log10(p_ch1):.2f} dB)")
print(f"   Channel 2 Power:  {p_ch2:.3e} ({10*np.log10(p_ch2):.2f} dB)")
print(f"   S_max (Maximized Direct Signal): {p_max:.3e} ({10*np.log10(p_max):.2f} dB) -> Gain: {direct_gain_db:+.2f} dB")
print(f"   S_null (Spatial Transmitter Null): {p_null:.3e} ({10*np.log10(p_null):.2f} dB) -> Spatial Null Depth: {null_depth_db:.2f} dB")

# 6. GPU Radar Correlations for Different Signal Combinations
pr_gpu = PassiveRadarGPU(device=device)

sa_gpu = torch.tensor(sa_dec, dtype=torch.complex64, device=device)
sb_gpu = torch.tensor(sb_dec, dtype=torch.complex64, device=device)
s_max_gpu = torch.tensor(s_max, dtype=torch.complex64, device=device)
s_null_gpu = torch.tensor(s_null, dtype=torch.complex64, device=device)

combinations = [
    {
        "name": "Combo A: Ref=S_max, Sur=Ch2 (Raw)",
        "ref": s_max_gpu, "sur": sb_gpu,
        "plot_name": "rdm_combo_a_max_vs_raw.png"
    },
    {
        "name": "Combo B: Ref=S_max, Sur=S_null (Max Ref vs Spatial Nulled Sur)",
        "ref": s_max_gpu, "sur": s_null_gpu,
        "plot_name": "rdm_combo_b_max_vs_null.png"
    },
    {
        "name": "Combo C: Ref=Ch1, Sur=S_null (Raw Ref vs Spatial Nulled Sur)",
        "ref": sa_gpu, "sur": s_null_gpu,
        "plot_name": "rdm_combo_c_ch1_vs_null.png"
    }
]

print("\n4. Calculating Radar Cross-Ambiguity / Correlation Maps on GPU...")
os.makedirs("output_plots", exist_ok=True)

import matplotlib.pyplot as plt

for combo in combinations:
    # ECA Clutter Cancellation on Surveillance signal
    sur_clean_gpu = pr_gpu.eca_clutter_cancellation_gpu(combo["ref"], combo["sur"], max_delay_taps=40)

    # Compute Range Doppler Map
    rdm_db_np, r_km, d_hz, v_kmh = pr_gpu.compute_caf_gpu(
        combo["ref"], sur_clean_gpu, target_fs, fc_effective, max_range_km=max_range_km, max_doppler_hz=400.0
    )

    # Detect targets
    noise_floor_per_doppler = np.median(rdm_db_np, axis=0, keepdims=True)
    rdm_norm_db = rdm_db_np - noise_floor_per_doppler
    rdm_norm_db -= np.max(rdm_norm_db)

    dets = pr.cfar_2d(
        rdm_norm_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=12.0,
        exclude_zero_doppler_hz=20.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
    )
    aircraft_dets = [d for d in dets if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

    plot_path = os.path.join("output_plots", combo["plot_name"])
    pr.plot_range_doppler_map(
        rdm_norm_db, r_km, v_kmh, d_hz, detections=aircraft_dets,
        title=f"{combo['name']} (Null Depth {null_depth_db:.1f} dB)",
        save_path=plot_path
    )
    print(f"   -> {combo['name']}: Saved plot to {plot_path} | Targets: {len(aircraft_dets)}")

print("\n=======================================================================")
print("PHASED NULLING RADAR PROCESSING COMPLETE!")
print("=======================================================================")
