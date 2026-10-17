"""
Full Recording Integration Processor (Non-Coherent & Max-Hold Accumulation).

Integrates all time frames of the recording into a single high-sensitivity Range-Doppler Map.
Suppresses noise floor by sqrt(M) and reveals faint aircraft targets.
"""

import os
import sys
import gc
import time
import numpy as np
import scipy.signal as signal

from iq_reader import IQReader, load_coherent_pair
import passive_radar as pr

file_a = "ch1_Fs102MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs102MHz_Fc640kHz_FMradio"

fc_raw = 102.0e6
fs_raw = 640000.0

cpi_sec = 4.0
step_sec = 2.0        # 50% overlap sliding window for maximum temporal gain
target_bw = 200000.0  # 200 kHz FM channel
target_fs = 200000.0  # 200 kHz sample rate

reader = IQReader(file_a)
total_sec = reader.get_duration_seconds(fs_raw)
total_samples = reader.total_samples

print("=======================================================================")
print(f"FULL RECORDING INTEGRATION ({total_sec:.2f} seconds total capture)")
print("=======================================================================")
print(f"  CPI Duration: {cpi_sec:.1f} s | Step: {step_sec:.1f} s (50% Overlap)")
print(f"  Target Bandwidth: {target_bw/1e3:.0f} kHz | Decimated Fs: {target_fs/1e3:.0f} kHz")
print("=======================================================================\n")

num_steps = int((total_sec - cpi_sec) / step_sec) + 1
print(f"Integrating {num_steps} sliding frames across the entire recording...\n")

# Detect FM carrier peak on first block
sa_init, sb_init = load_coherent_pair(file_a, file_b, num_samples=32768, offset_samples=0)
freqs, psd_a = signal.welch(sa_init, fs=fs_raw, nperseg=2048, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd_a = np.fft.fftshift(psd_a)
peak_offset_hz = freqs[np.argmax(psd_a)]
fc_effective = fc_raw + peak_offset_hz
del sa_init, sb_init
gc.collect()

print(f"Detected FM Carrier Offset: {peak_offset_hz/1e3:+.2f} kHz (Tuned Fc: {fc_effective/1e6:.4f} MHz)")

# Accumulators for Non-Coherent Integration (NCI) and Max-Hold
rdm_nci_linear = None
rdm_max_linear = None
axes_info = None

t_start_all = time.time()

for step in range(num_steps):
    offset_sec = step * step_sec
    offset_samples = int(offset_sec * fs_raw)
    num_raw = int(cpi_sec * fs_raw)

    if offset_samples + num_raw > total_samples:
        break

    # 1. Load slice
    sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw, offset_samples=offset_samples)

    # 2. In-place Frequency Shift, Low-pass Filter & Decimate to 200 kHz
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

    # 3. Phased Array Beamforming (using 135° for max DPI nulling)
    phase_deg = 135.0
    s_sum = pr.beamform_sum(sa_dec, sb_dec, phase_deg)
    s_ref, s_sur = pr.select_channels(sa_dec, sb_dec, s_sum, "SUM", "B")

    # 4. ECA Clutter Cancellation
    s_sur_clean = pr.eca_clutter_cancellation(s_ref, s_sur, max_delay_taps=40)

    # 5. Range-Doppler Map
    rdm_db, r_km, d_hz, v_kmh = pr.compute_cross_ambiguity_function(
        s_ref, s_sur_clean, target_fs, fc_effective, max_range_km=80.0, max_doppler_hz=400.0
    )

    # Convert dB to linear power for integration
    power_lin = 10.0 ** (rdm_db / 10.0)

    if rdm_nci_linear is None:
        rdm_nci_linear = np.copy(power_lin)
        rdm_max_linear = np.copy(power_lin)
        axes_info = (r_km, d_hz, v_kmh)
    else:
        rdm_nci_linear += power_lin
        rdm_max_linear = np.maximum(rdm_max_linear, power_lin)

    del sa_dec, sb_dec, s_sum, s_ref, s_sur, s_sur_clean, rdm_db, power_lin
    gc.collect()

    if (step + 1) % 5 == 0 or step == num_steps - 1:
        print(f"  Integrated frame {step+1}/{num_steps} (Time: {offset_sec+cpi_sec:.1f}s / {total_sec:.1f}s)")

# Compute final integrated dB maps
rdm_nci_db = 10.0 * np.log10(rdm_nci_linear / float(num_steps))
rdm_nci_db -= np.max(rdm_nci_db)

rdm_max_db = 10.0 * np.log10(rdm_max_linear)
rdm_max_db -= np.max(rdm_max_db)

r_km, d_hz, v_kmh = axes_info

print("\n=======================================================================")
print(f"INTEGRATION COMPLETE ({num_steps} frames integrated in {time.time()-t_start_all:.1f}s)")
print("=======================================================================")

# Run 2D CFAR Detector on Non-Coherent Integrated Picture
print("Running 2D CFAR Detector on Integrated Range-Doppler Map...")
dets_nci = pr.cfar_2d(
    rdm_nci_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=12.0,
    exclude_zero_doppler_hz=20.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
)

# Exclude audio subcarriers
aircraft_nci = [d for d in dets_nci if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

print("-" * 70)
print(f"INTEGRATED RADAR PICTURE TARGET SUMMARY ({len(aircraft_nci)} target candidates):")
if aircraft_nci:
    print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
    print("-" * 68)
    for idx, det in enumerate(aircraft_nci, 1):
        print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
else:
    print("  No aircraft targets detected above CFAR threshold in the integrated picture.")
print("-" * 70)

# Save Integrated Radar Plots
os.makedirs("output_plots", exist_ok=True)

nci_plot_path = "output_plots/ch1_ch2_integrated_nci_rdm.png"
pr.plot_range_doppler_map(
    rdm_nci_db, r_km, v_kmh, d_hz, detections=aircraft_nci,
    title=f"Non-Coherent Integrated Range-Doppler Map (All {total_sec:.1f}s Capture)",
    save_path=nci_plot_path
)
print(f"Saved Non-Coherent Integrated RDM plot -> {nci_plot_path}")

max_plot_path = "output_plots/ch1_ch2_integrated_max_rdm.png"
pr.plot_range_doppler_map(
    rdm_max_db, r_km, v_kmh, d_hz, detections=aircraft_nci,
    title=f"Max-Hold Integrated Range-Doppler Map (All {total_sec:.1f}s Capture)",
    save_path=max_plot_path
)
print(f"Saved Max-Hold Integrated RDM plot -> {max_plot_path}")
print("=======================================================================")
