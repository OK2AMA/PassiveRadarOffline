"""
GPU-Accelerated Passive Radar Processor & Sensitivity Enhancer.

Uses NVIDIA GeForce RTX 2060 GPU (PyTorch CUDA) for:
1. Fast Phase Angle & Magnitude Spatial Beamforming Sweeps
2. GPU ECA-B Clutter Cancellation
3. High-Speed 2D FFT Range-Doppler Map (CAF) Generation
4. Full Recording GPU Non-Coherent & Max-Hold Integration
5. Subspace Background Normalization for Faint Target Visibility
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

file_a = "ch1_Fs102MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs102MHz_Fc640kHz_FMradio"

fc_raw = 102.0e6
fs_raw = 640000.0

cpi_sec = 4.0
step_sec = 2.0
target_bw = 200000.0
target_fs = 200000.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=======================================================================")
print(f"GPU PASSIVE RADAR PROCESSOR ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
print("=======================================================================")

reader = IQReader(file_a)
total_sec = reader.get_duration_seconds(fs_raw)
total_samples = reader.total_samples
num_steps = int((total_sec - cpi_sec) / step_sec) + 1

print(f"  Capture Duration: {total_sec:.2f} s ({num_steps} overlapping frames)")
print(f"  GPU Device: {device.upper()} | Decimated Fs: {target_fs/1e3:.0f} kHz")
print("=======================================================================\n")

# Detect FM carrier peak
sa_init, sb_init = load_coherent_pair(file_a, file_b, num_samples=32768, offset_samples=0)
freqs, psd_a = signal.welch(sa_init, fs=fs_raw, nperseg=2048, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd_a = np.fft.fftshift(psd_a)
peak_offset_hz = freqs[np.argmax(psd_a)]
fc_effective = fc_raw + peak_offset_hz
del sa_init, sb_init
gc.collect()

print(f"1. Detected FM station peak offset: {peak_offset_hz/1e3:+.2f} kHz (Tuned Fc: {fc_effective/1e6:.4f} MHz)")

pr_gpu = PassiveRadarGPU(device=device)

# GPU accumulators
gpu_nci_power = None
gpu_max_power = None
axes_info = None

t_start_gpu = time.time()

print("2. Processing 38 sliding frames on GPU VRAM...")

for step in range(num_steps):
    offset_sec = step * step_sec
    offset_samples = int(offset_sec * fs_raw)
    num_raw = int(cpi_sec * fs_raw)

    if offset_samples + num_raw > total_samples:
        break

    # Load raw slice
    sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw, offset_samples=offset_samples)

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

    # Move tensors to GPU
    sa_gpu = torch.tensor(sa_dec, dtype=torch.complex64, device=device)
    sa_gpu -= torch.mean(sa_gpu)

    sb_gpu = torch.tensor(sb_dec, dtype=torch.complex64, device=device)
    sb_gpu -= torch.mean(sb_gpu)
    del sa_dec, sb_dec

    # GPU Phased Array Beamforming (135° phase offset)
    s_sum_gpu = pr_gpu.beamform_sum_gpu(sa_gpu, sb_gpu, phase_offset_deg=135.0)

    # GPU ECA Clutter Cancellation (40 taps)
    s_sur_clean_gpu = pr_gpu.eca_clutter_cancellation_gpu(s_sum_gpu, sb_gpu, max_delay_taps=40)

    # GPU Range-Doppler Map (CAF)
    rdm_db_np, r_km, d_hz, v_kmh = pr_gpu.compute_caf_gpu(
        s_sum_gpu, s_sur_clean_gpu, target_fs, fc_effective, max_range_km=80.0, max_doppler_hz=400.0
    )

    rdm_power_gpu = torch.tensor(10.0 ** (rdm_db_np / 10.0), device=device)

    if gpu_nci_power is None:
        gpu_nci_power = rdm_power_gpu.clone()
        gpu_max_power = rdm_power_gpu.clone()
        axes_info = (r_km, d_hz, v_kmh)
    else:
        gpu_nci_power += rdm_power_gpu
        gpu_max_power = torch.maximum(gpu_max_power, rdm_power_gpu)

    del sa_gpu, sb_gpu, s_sum_gpu, s_sur_clean_gpu, rdm_power_gpu
    torch.cuda.empty_cache()

    if (step + 1) % 10 == 0 or step == num_steps - 1:
        print(f"   Processed GPU frame {step+1}/{num_steps} ({offset_sec+cpi_sec:.1f}s / {total_sec:.1f}s)")

# Compute final integrated dB maps on GPU
gpu_nci_db = 10.0 * torch.log10(gpu_nci_power / float(num_steps))
gpu_nci_db -= torch.max(gpu_nci_db)
rdm_nci_np = gpu_nci_db.cpu().numpy()

gpu_max_db = 10.0 * torch.log10(gpu_max_power)
gpu_max_db -= torch.max(gpu_max_db)
rdm_max_np = gpu_max_db.cpu().numpy()

total_gpu_time = time.time() - t_start_gpu
print(f"\n=======================================================================")
print(f"GPU INTEGRATION COMPLETE: 38 frames processed in {total_gpu_time:.2f} seconds!")
print("=======================================================================")

r_km, d_hz, v_kmh = axes_info

# Background Normalization (Reciprocal noise floor subtraction) to enhance target contrast
noise_floor_per_doppler = np.median(rdm_nci_np, axis=0, keepdims=True)
rdm_normalized_db = rdm_nci_np - noise_floor_per_doppler
rdm_normalized_db -= np.max(rdm_normalized_db)

# Run CFAR Target Detector on normalized GPU picture
dets = pr.cfar_2d(
    rdm_normalized_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=11.5,
    exclude_zero_doppler_hz=20.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
)

aircraft_dets = [d for d in dets if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

print("-" * 70)
print(f"GPU ENHANCED TARGET SUMMARY ({len(aircraft_dets)} targets detected):")
if aircraft_dets:
    print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
    print("-" * 68)
    for idx, det in enumerate(aircraft_dets, 1):
        print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
else:
    print("  No aircraft targets detected above CFAR threshold.")
print("-" * 70)

# Save GPU Integrated Radar Plots
os.makedirs("output_plots", exist_ok=True)

gpu_nci_path = "output_plots/ch1_ch2_gpu_integrated_rdm.png"
pr.plot_range_doppler_map(
    rdm_normalized_db, r_km, v_kmh, d_hz, detections=aircraft_dets,
    title=f"GPU Enhanced Range-Doppler Map (NVIDIA RTX 2060, {total_sec:.1f}s Integration)",
    save_path=gpu_nci_path
)
print(f"Saved GPU Enhanced RDM plot -> {gpu_nci_path}")
print("=======================================================================")
