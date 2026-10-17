"""
Passive Radar Processor for 95.1 MHz FM Recording.

Settings requested by user:
- Clutter cancellation filter: DISABLED (s_sur_clean = s_sur)
- Max Range: 150 km
- Full 145.72s integration on ch1_Fs95.1MHz_Fc640kHz_FMradio & ch2_Fs95.1MHz_Fc640kHz_FMradio
"""

import os
import sys
import gc
import time
import numpy as np
import scipy.signal as signal
import torch

from iq_reader import IQReader, load_coherent_pair
from passive_radar_gpu import PassiveRadarGPU
import passive_radar as pr

file_a = "ch1_Fs95.1MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs95.1MHz_Fc640kHz_FMradio"

fc_raw = 95.1e6
fs_raw = 640000.0

cpi_sec = 4.0
step_sec = 2.0
target_bw = 200000.0
target_fs = 200000.0
max_range_km = 150.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=======================================================================")
print(f"95.1 MHz PASSIVE RADAR: NO CLUTTER FILTER | MAX RANGE {max_range_km:.0f} KM")
print("=======================================================================")

reader = IQReader(file_a)
total_sec = reader.get_duration_seconds(fs_raw)
total_samples = reader.total_samples
num_steps = int((total_sec - cpi_sec) / step_sec) + 1

print(f"  Files: {file_a} & {file_b}")
print(f"  Total Capture Duration: {total_sec:.2f} s ({num_steps} overlapping frames)")
print(f"  ECA Clutter Filter: DISABLED")
print(f"  Max Range: {max_range_km:.0f} km")
print(f"  GPU Device: {device.upper()}")
print("=======================================================================\n")

# Detect FM peak
sa_init, sb_init = load_coherent_pair(file_a, file_b, num_samples=32768, offset_samples=0)
freqs, psd_a = signal.welch(sa_init, fs=fs_raw, nperseg=2048, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd_a = np.fft.fftshift(psd_a)
peak_offset_hz = freqs[np.argmax(psd_a)]
fc_effective = fc_raw + peak_offset_hz
del sa_init, sb_init
gc.collect()

print(f"1. Detected FM carrier offset: {peak_offset_hz/1e3:+.2f} kHz (Tuned Fc: {fc_effective/1e6:.4f} MHz)")

pr_gpu = PassiveRadarGPU(device=device)

gpu_nci_power = None
axes_info = None

t_start_gpu = time.time()
print(f"2. Processing {num_steps} sliding frames across full recording...")

for step in range(num_steps):
    offset_sec = step * step_sec
    offset_samples = int(offset_sec * fs_raw)
    num_raw = int(cpi_sec * fs_raw)

    if offset_samples + num_raw > total_samples:
        break

    # Load raw slice
    sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw, offset_samples=offset_samples)

    # Shift, LPF, Decimate to 200 kHz
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

    # Tensors to GPU
    sa_gpu = torch.tensor(sa_dec, dtype=torch.complex64, device=device)
    sa_gpu -= torch.mean(sa_gpu)

    sb_gpu = torch.tensor(sb_dec, dtype=torch.complex64, device=device)
    sb_gpu -= torch.mean(sb_gpu)
    del sa_dec, sb_dec

    # Phased Array Beamforming
    s_sum_gpu = pr_gpu.beamform_sum_gpu(sa_gpu, sb_gpu, phase_offset_deg=135.0)

    # CLUTTER CANCELLATION IS DISABLED AS REQUESTED:
    s_sur_clean_gpu = sb_gpu  # Raw surveillance channel without ECA clutter filter

    # GPU Range-Doppler Map (max_range_km = 150 km)
    rdm_db_np, r_km, d_hz, v_kmh = pr_gpu.compute_caf_gpu(
        s_sum_gpu, s_sur_clean_gpu, target_fs, fc_effective, max_range_km=max_range_km, max_doppler_hz=400.0
    )

    rdm_power_gpu = torch.tensor(10.0 ** (rdm_db_np / 10.0), device=device)

    if gpu_nci_power is None:
        gpu_nci_power = rdm_power_gpu.clone()
        axes_info = (r_km, d_hz, v_kmh)
    else:
        gpu_nci_power += rdm_power_gpu

    del sa_gpu, sb_gpu, s_sum_gpu, s_sur_clean_gpu, rdm_power_gpu
    torch.cuda.empty_cache()

    if (step + 1) % 15 == 0 or step == num_steps - 1:
        print(f"   Processed frame {step+1}/{num_steps} ({offset_sec+cpi_sec:.1f}s / {total_sec:.1f}s)")

gpu_nci_db = 10.0 * torch.log10(gpu_nci_power / float(num_steps))
gpu_nci_db -= torch.max(gpu_nci_db)
rdm_nci_np = gpu_nci_db.cpu().numpy()

total_gpu_time = time.time() - t_start_gpu
print(f"\n=======================================================================")
print(f"INTEGRATION COMPLETE: {num_steps} frames processed in {total_gpu_time:.2f} seconds!")
print("=======================================================================")

r_km, d_hz, v_kmh = axes_info

# Background normalization for visualization
noise_floor_per_doppler = np.median(rdm_nci_np, axis=0, keepdims=True)
rdm_normalized_db = rdm_nci_np - noise_floor_per_doppler
rdm_normalized_db -= np.max(rdm_normalized_db)

# 2D CFAR Detector
dets = pr.cfar_2d(
    rdm_normalized_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=12.0,
    exclude_zero_doppler_hz=20.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
)

aircraft_dets = [d for d in dets if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

print("-" * 70)
print(f"AIRPLANE SUMMARY (95.1 MHz, No Clutter Filter, 150 km, {len(aircraft_dets)} targets):")
if aircraft_dets:
    print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
    print("-" * 68)
    for idx, det in enumerate(aircraft_dets, 1):
        print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
else:
    print("  No aircraft targets detected above CFAR threshold.")
print("-" * 70)

# Save Plot
os.makedirs("output_plots", exist_ok=True)
noclutter_plot_path = "output_plots/ch1_ch2_95.1mhz_noclutter_150km_rdm.png"
pr.plot_range_doppler_map(
    rdm_normalized_db, r_km, v_kmh, d_hz, detections=aircraft_dets,
    title=f"Range-Doppler Map 95.1 MHz (No Clutter Filter, Range 150 km, {total_sec:.1f}s)",
    save_path=noclutter_plot_path
)
print(f"Saved 150 km No-Clutter Range-Doppler Map plot -> {noclutter_plot_path}")
print("=======================================================================")
