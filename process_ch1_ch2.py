"""
Passive Radar Processor for ch1_Fs102MHz_Fc640kHz_FMradio & ch2_Fs102MHz_Fc640kHz_FMradio.

Features:
- RAM-Safe chunked loading & in-place operations
- FM carrier offset detection & tuning to 0 Hz
- 100 kHz low-pass filtering & 200 kHz decimation
- Phased array beamforming (channel sum with phase offset)
- Long Coherent Processing Interval (CPI) integration
- ECA-B direct path & clutter cancellation
- High-speed Range-Doppler Map (CAF) & 2D CFAR airplane detection
"""

import os
import sys
import gc
import time
import numpy as np
import scipy.signal as signal

from iq_reader import load_coherent_pair
import passive_radar as pr

file_a = "ch1_Fs102MHz_Fc640kHz_FMradio"
file_b = "ch2_Fs102MHz_Fc640kHz_FMradio"

fc_raw = 102.0e6
fs_raw = 640000.0

cpi_sec = 4.0        # Long coherent integration (4 seconds)
target_bw = 200000.0 # 200 kHz FM channel bandwidth
target_fs = 200000.0 # Decimated sample rate 200 kHz

print("=======================================================================")
print("PASSIVE RADAR ANALYSIS FOR CH1 & CH2 (102 MHz FM, 640 kHz Fs)")
print("=======================================================================")
print(f"  Files: {file_a} & {file_b}")
print(f"  Raw Fs: {fs_raw/1e3:.0f} kHz | Raw Fc: {fc_raw/1e6:.1f} MHz")
print(f"  Target Bandwidth: {target_bw/1e3:.0f} kHz | Decimated Fs: {target_fs/1e3:.0f} kHz")
print(f"  CPI Integration Time: {cpi_sec:.1f} seconds")
print("=======================================================================\n")

# 1. Load initial CPI block safely
num_raw_samples = int(cpi_sec * fs_raw)
print(f"1. Reading {cpi_sec:.1f} seconds of 16I16Q IQ data ({num_raw_samples} samples per channel)...")
sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw_samples, offset_samples=0)
print(f"   Raw array RAM footprint: {(len(sa_raw) + len(sb_raw))*8 / 1e6:.2f} MB")

# 2. Detect FM carrier peak
print("2. Detecting dominant FM radio station carrier frequency...")
sub_len = min(len(sa_raw), 32768)
freqs, psd_a = signal.welch(sa_raw[:sub_len], fs=fs_raw, nperseg=2048, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd_a = np.fft.fftshift(psd_a)
peak_offset_hz = freqs[np.argmax(psd_a)]
fc_effective = fc_raw + peak_offset_hz

print(f"   Detected FM carrier peak offset: {peak_offset_hz/1e3:+.2f} kHz")
print(f"   Tuned FM station carrier frequency: {fc_effective/1e6:.4f} MHz")

# 3. Frequency shift to DC, 100 kHz LPF, and 200 kHz Decimation
print("3. Tuning FM station to 0 Hz, filtering to 200 kHz BW & decimating...")
t = np.arange(len(sa_raw), dtype=np.float32) / float(fs_raw)
shift_vec = np.exp(-1j * (2.0 * np.pi * peak_offset_hz) * t)

sa_tuned = sa_raw * shift_vec
sb_tuned = sb_raw * shift_vec
del sa_raw, sb_raw, shift_vec, t
gc.collect()

# Lowpass FIR filter (cutoff = 100 kHz)
cutoff_norm = (target_bw / 2.0) / (fs_raw / 2.0)
fir_coeff = signal.firwin(61, cutoff_norm, window='hamming')

sa_filt = signal.lfilter(fir_coeff, 1.0, sa_tuned)
sb_filt = signal.lfilter(fir_coeff, 1.0, sb_tuned)
del sa_tuned, sb_tuned
gc.collect()

# Decimate 640 kHz -> 200 kHz (polyphase resampling up=1, down=3.2 -> up=5, down=16)
sa_dec = signal.resample_poly(sa_filt, 5, 16).astype(np.complex64)
sb_dec = signal.resample_poly(sb_filt, 5, 16).astype(np.complex64)
del sa_filt, sb_filt
gc.collect()

print(f"   Decimated sample count: {len(sa_dec)} samples at Fs = {target_fs/1e3:.0f} kHz")
print(f"   Optimized RAM footprint: {(len(sa_dec) + len(sb_dec))*8 / 1e6:.2f} MB")

# Save Spectrum Plot
os.makedirs("output_plots", exist_ok=True)
s_sum_0 = pr.beamform_sum(sa_dec, sb_dec, 0.0)
pr.plot_channels_and_spectra(sa_dec, sb_dec, s_sum_0, target_fs,
                             title="Tuned & Decimated (200 kHz) Channel Spectra",
                             save_path="output_plots/ch1_ch2_spectra_200khz.png")
print("   Saved spectrum plot -> output_plots/ch1_ch2_spectra_200khz.png")

# 4. Reference Autocorrelation
print("4. Computing Reference Channel Autocorrelation...")
lags, delays_us, equiv_range_km, r_norm = pr.compute_autocorrelation(s_sum_0, target_fs, max_lags=300)
pr.plot_autocorrelation(lags, delays_us, equiv_range_km, r_norm,
                        title="Reference Signal Autocorrelation (200 kHz FM Channel)",
                        save_path="output_plots/ch1_ch2_autocorrelation_200khz.png")
print("   Saved autocorrelation plot -> output_plots/ch1_ch2_autocorrelation_200khz.png")

# 5. Phased Array Sweep & ECA Clutter Cancellation
print("\n5. Running Phased Array Phase Sweep & ECA Clutter Cancellation...")
best_dets = []
best_phase = 0.0
best_rdm_db = None
best_axes = None

for phase_deg in [0.0, 45.0, 90.0, 135.0, 180.0]:
    t_start = time.time()
    s_sum = pr.beamform_sum(sa_dec, sb_dec, phase_deg)
    s_ref, s_sur = pr.select_channels(sa_dec, sb_dec, s_sum, "SUM", "B")

    # ECA Clutter Cancellation (40 taps at 200 kHz = 40 * 1500m = 60 km range span)
    s_sur_clean = pr.eca_clutter_cancellation(s_ref, s_sur, max_delay_taps=40)
    att_db = 10 * np.log10(np.mean(np.abs(s_sur)**2) / (np.mean(np.abs(s_sur_clean)**2) + 1e-12))

    # Fast Strided Range-Doppler Map
    rdm_db, r_km, d_hz, v_kmh = pr.compute_cross_ambiguity_function(
        s_ref, s_sur_clean, target_fs, fc_effective, max_range_km=80.0, max_doppler_hz=400.0
    )

    # 2D CFAR Detector
    dets = pr.cfar_2d(
        rdm_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=14.0,
        exclude_zero_doppler_hz=15.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
    )

    # Exclude audio subcarrier sidebands around ~334 Hz and ~187 Hz
    aircraft_dets = [d for d in dets if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

    proc_time = time.time() - t_start
    print(f"   Phase {phase_deg:5.1f}° | DPI Attenuation: {att_db:5.2f} dB | Proc Time: {proc_time*1000:5.1f} ms | Targets: {len(aircraft_dets)}")

    if len(aircraft_dets) > len(best_dets) or best_rdm_db is None:
        best_dets = aircraft_dets
        best_phase = phase_deg
        best_rdm_db = rdm_db
        best_axes = (r_km, d_hz, v_kmh)

print("-" * 70)
print(f"AIRPLANE TARGET DETECTION SUMMARY (Best Phased Array Angle: {best_phase:.0f}°):")
if best_dets:
    print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
    print("-" * 68)
    for idx, det in enumerate(best_dets, 1):
        print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
else:
    print("   No aircraft targets detected above CFAR threshold in this 4.0s CPI block.")
print("-" * 70)

# Save Range Doppler Map Plot
r_km, d_hz, v_kmh = best_axes
rdm_plot_path = "output_plots/ch1_ch2_range_doppler_map.png"
pr.plot_range_doppler_map(
    best_rdm_db, r_km, v_kmh, d_hz, detections=best_dets,
    title=f"Range-Doppler Map (ch1 & ch2, 4.0s CPI, Decimated 200 kHz, Phase {best_phase:.0f}°)",
    save_path=rdm_plot_path
)
print(f"Saved Range-Doppler Map plot -> {rdm_plot_path}")
print("=======================================================================")
