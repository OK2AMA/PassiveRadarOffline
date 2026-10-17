"""
DVB-T Phased Array & Spatial Nulling Passive Radar Processor (Fc 514 MHz, Fs 3 MHz).

Processes: IQ/dvbt_1/a514_3m & IQ/dvbt_1/b514_3m
1. Cross-correlation phase difference & coherence measurement.
2. Direct signal maximization S_max (+3 dB gain).
3. Direct signal spatial nulling S_null.
4. GPU-accelerated high-resolution DVB-T Range-Doppler Map (100m range resolution).
5. 2D CFAR aircraft target detection.
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

file_a = "Ch1_Fc514MHz_Fs3MHz_DVBT"
file_b = "Ch2_Fc514MHz_Fs3MHz_DVBT"

fc = 514.0e6
fs = 3.0e6
cpi_sec = 2.0  # 2.0s CPI = 6,000,000 samples

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=======================================================================")
print(f"DVB-T PHASED ARRAY & SPATIAL NULLING RADAR (Fc 514 MHz, Fs 3.0 MHz)")
print("=======================================================================")

# 1. Load 2.0s CPI
num_samples = int(cpi_sec * fs)
sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_samples, offset_samples=0)

# DC removal
sa_raw -= np.mean(sa_raw)
sb_raw -= np.mean(sb_raw)

# 2. Calculate Cross-Correlation between Channel A and Channel B
cross_corr = np.sum(sa_raw * np.conj(sb_raw))
delta_phi_rad = np.angle(cross_corr)
delta_phi_deg = np.degrees(delta_phi_rad)

p1 = np.mean(np.abs(sa_raw)**2)
p2 = np.mean(np.abs(sb_raw)**2)
coherence = np.abs(cross_corr) / (np.sqrt(np.sum(np.abs(sa_raw)**2)) * np.sqrt(np.sum(np.abs(sb_raw)**2)))
alpha = np.sqrt(p1 / p2)

print("\n1. DVB-T Cross-Channel Correlation Results:")
print(f"   Direct Path Phase Offset delta_phi: {delta_phi_deg:+.2f}° ({delta_phi_rad:+.4f} rad)")
print(f"   Cross-Channel Coherence Magnitude: {coherence:.4f}")
print(f"   Channel A Power: {p1:.3e} ({10*np.log10(p1):.2f} dB)")
print(f"   Channel B Power: {p2:.3e} ({10*np.log10(p2):.2f} dB)")

# 3. Synthesize S_max and S_null
s_max = sa_raw + alpha * sb_raw * np.exp(-1j * delta_phi_rad)
s_null = sa_raw - alpha * sb_raw * np.exp(-1j * delta_phi_rad)

p_max = np.mean(np.abs(s_max)**2)
p_null = np.mean(np.abs(s_null)**2)
direct_gain_db = 10.0 * np.log10(p_max / max(p1, p2))
null_depth_db = 10.0 * np.log10(p_null / p_max)

print(f"\n2. Phased Array Synthesis Performance:")
print(f"   S_max (Maximized DVB-T Direct Signal): {p_max:.3e} ({10*np.log10(p_max):.2f} dB) -> Gain: {direct_gain_db:+.2f} dB")
print(f"   S_null (Spatial Transmitter Null):    {p_null:.3e} ({10*np.log10(p_null):.2f} dB) -> Null Depth: {null_depth_db:.2f} dB")

# 4. GPU Range-Doppler Map Processing
pr_gpu = PassiveRadarGPU(device=device)

sa_gpu = torch.tensor(sa_raw, dtype=torch.complex64, device=device)
sb_gpu = torch.tensor(sb_raw, dtype=torch.complex64, device=device)
s_max_gpu = torch.tensor(s_max, dtype=torch.complex64, device=device)
s_null_gpu = torch.tensor(s_null, dtype=torch.complex64, device=device)

del sa_raw, sb_raw, s_max, s_null
gc.collect()

# ECA Clutter Cancellation on Surveillance Channel
s_sur_clean_gpu = pr_gpu.eca_clutter_cancellation_gpu(s_max_gpu, s_null_gpu, max_delay_taps=60)

print("\n3. Computing High-Resolution DVB-T Range-Doppler Map on GPU VRAM...")
rdm_db_np, r_km, d_hz, v_kmh = pr_gpu.compute_caf_gpu(
    s_max_gpu, s_sur_clean_gpu, fs, fc, max_range_km=50.0, max_doppler_hz=500.0
)

# Background Normalization
noise_floor_per_doppler = np.median(rdm_db_np, axis=0, keepdims=True)
rdm_norm_db = rdm_db_np - noise_floor_per_doppler
rdm_norm_db -= np.max(rdm_norm_db)

# 2D CFAR Detector
dets = pr.cfar_2d(
    rdm_norm_db, guard_cells=(4, 6), ref_cells=(16, 24), threshold_db=14.0,
    exclude_zero_doppler_hz=15.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
)

print("-" * 70)
print(f"DVB-T RADAR TARGET SUMMARY ({len(dets)} target candidates detected):")
if dets:
    print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
    print("-" * 68)
    for idx, det in enumerate(dets, 1):
        print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
else:
    print("  No aircraft targets detected above CFAR threshold.")
print("-" * 70)

# Save Annotated Plot
os.makedirs("output_plots", exist_ok=True)
plot_path = "output_plots/dvbt_phased_null_annotated_rdm.png"

metadata = {
    "fc_mhz": fc / 1e6,
    "fs_mhz": fs / 1e6,
    "range_res_m": 299792458.0 / fs,
    "cpi_sec": cpi_sec,
    "doppler_res_hz": 1.0 / cpi_sec,
    "phase_deg": delta_phi_deg,
    "phase_rad": delta_phi_rad,
    "coherence": coherence,
    "p_ch1_db": 10 * np.log10(p1),
    "p_ch2_db": 10 * np.log10(p2),
    "direct_gain_db": direct_gain_db,
    "null_depth_db": null_depth_db
}

pr.plot_annotated_range_doppler_map(
    rdm_norm_db, r_km, v_kmh, d_hz,
    metadata=metadata,
    detections=dets,
    title=f"DVB-T Passive Radar Map (Fc 514 MHz, Fs 3 MHz, 100m Resolution)",
    save_path=plot_path
)
print(f"Saved Annotated DVB-T Range-Doppler Map plot -> {plot_path}")
print("=======================================================================")
