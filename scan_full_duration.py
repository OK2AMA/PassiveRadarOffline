"""
Full 78.8-second RAM-Safe Sliding Analysis for ch1 & ch2.
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
step_sec = 4.0
target_bw = 200000.0
target_fs = 200000.0

reader = IQReader(file_a)
total_sec = reader.get_duration_seconds(fs_raw)
total_samples = reader.total_samples

print("=======================================================================")
print(f"SLIDING SEARCH ACROSS FULL RECORDING ({total_sec:.2f} seconds)")
print("=======================================================================\n")

num_steps = int((total_sec - cpi_sec) / step_sec) + 1

all_target_events = []

for step in range(num_steps):
    offset_sec = step * step_sec
    offset_samples = int(offset_sec * fs_raw)
    num_raw = int(cpi_sec * fs_raw)
    
    if offset_samples + num_raw > total_samples:
        break

    # 1. Load slice
    sa_raw, sb_raw = load_coherent_pair(file_a, file_b, num_samples=num_raw, offset_samples=offset_samples)
    
    # 2. Detect peak offset on first slice
    if step == 0:
        sub_len = min(len(sa_raw), 32768)
        freqs, psd_a = signal.welch(sa_raw[:sub_len], fs=fs_raw, nperseg=2048, return_onesided=False)
        freqs = np.fft.fftshift(freqs)
        psd_a = np.fft.fftshift(psd_a)
        peak_offset_hz = freqs[np.argmax(psd_a)]
        fc_effective = fc_raw + peak_offset_hz
        print(f"Detected FM carrier offset: {peak_offset_hz/1e3:+.2f} kHz | Actual Fc: {fc_effective/1e6:.4f} MHz\n")

    # 3. Frequency shift, LPF, Decimate (in-place RAM safe)
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

    # 4. Phased array sweep (using 135 deg for max DPI attenuation)
    phase_deg = 135.0
    s_sum = pr.beamform_sum(sa_dec, sb_dec, phase_deg)
    s_ref, s_sur = pr.select_channels(sa_dec, sb_dec, s_sum, "SUM", "B")

    s_sur_clean = pr.eca_clutter_cancellation(s_ref, s_sur, max_delay_taps=40)

    rdm_db, r_km, d_hz, v_kmh = pr.compute_cross_ambiguity_function(
        s_ref, s_sur_clean, target_fs, fc_effective, max_range_km=80.0, max_doppler_hz=400.0
    )

    dets = pr.cfar_2d(
        rdm_db, guard_cells=(3, 5), ref_cells=(12, 20), threshold_db=13.5,
        exclude_zero_doppler_hz=20.0, doppler_axis=d_hz, range_axis=r_km, velocity_axis=v_kmh
    )

    # Exclude audio subcarriers
    aircraft_dets = [d for d in dets if abs(abs(d['doppler_hz']) - 334.0) > 15.0 and abs(abs(d['doppler_hz']) - 187.0) > 15.0]

    if aircraft_dets:
        print(f"Time [{offset_sec:5.1f}s - {offset_sec+cpi_sec:5.1f}s]: DETECTED {len(aircraft_dets)} candidate target(s):")
        for d in aircraft_dets:
            print(f"  -> Range: {d['range_km']:.2f} km | Velocity: {d['velocity_kmh']:+.1f} km/h (Doppler: {d['doppler_hz']:+.1f} Hz) | SNR: {d['snr_db']:.2f} dB")
            d["time_sec"] = offset_sec
            all_target_events.append(d)
    else:
        print(f"Time [{offset_sec:5.1f}s - {offset_sec+cpi_sec:5.1f}s]: No aircraft detections.")

    del sa_dec, sb_dec, s_sum, s_ref, s_sur, s_sur_clean, rdm_db
    gc.collect()

print("\n=======================================================================")
print(f"FULL SCAN COMPLETE: Total airplane target events found: {len(all_target_events)}")
print("=======================================================================")
