"""
Passive Radar Processor & IQ File Analyzer.

Usage:
  python3 run_analysis.py --config config.json
  python3 run_analysis.py --search-all
  python3 run_analysis.py --file_a IQ/a102_640 --file_b IQ/b102_640 --fc 102.0e6 --phase 45.0
"""

import argparse
import json
import os
import sys
import numpy as np

from iq_reader import IQReader, load_coherent_pair
import passive_radar as pr


def parse_args():
    parser = argparse.ArgumentParser(description="Passive Radar IQ Analyzer for AD9363 16I16Q Captures")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json file")
    parser.add_argument("--file_a", type=str, default=None, help="Channel A IQ file")
    parser.add_argument("--file_b", type=str, default=None, help="Channel B IQ file")
    parser.add_argument("--fs", type=float, default=None, help="Sampling frequency (Hz)")
    parser.add_argument("--fc", type=float, default=None, help="Center frequency (Hz)")
    parser.add_argument("--phase", type=float, default=None, help="Phased array phase offset (degrees)")
    parser.add_argument("--cpi", type=float, default=None, help="Coherent Processing Interval (seconds)")
    parser.add_argument("--search-all", action="store_true", help="Search all paired IQ files in IQ/ directory for aircraft")
    parser.add_argument("--show", action="store_true", help="Show interactive matplotlib plots")
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file '{config_path}' not found. Using defaults.")
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def analyze_dataset(file_a, file_b, fs_hz, fc_hz, phase_offset_deg=0.0, cpi_sec=1.0, offset_samples=0,
                    ref_source="SUM", sur_source="B", max_range_km=50.0, max_doppler_hz=500.0,
                    clutter_taps=40, cfar_thresh_db=12.0, output_dir="output_plots", prefix="", show_plots=False):
    """
    Run full Passive Radar pipeline on a dual-channel IQ dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(f"PASSIVE RADAR PROCESSING: {os.path.basename(file_a)} & {os.path.basename(file_b)}")
    print(f"  Fc: {fc_hz/1e6:.2f} MHz | Fs: {fs_hz/1e3:.1f} kHz | CPI: {cpi_sec:.2f} s")
    print(f"  Phased Array Offset: {phase_offset_deg:.1f} deg | Ref Source: {ref_source} | Sur Source: {sur_source}")
    print("=" * 70)

    num_samples = int(cpi_sec * fs_hz)

    # 1. Load IQ data
    print("1. Loading 16I16Q dual-channel IQ samples...")
    sa, sb = load_coherent_pair(file_a, file_b, num_samples=num_samples, offset_samples=offset_samples)
    print(f"   Loaded {len(sa)} complex samples ({len(sa)/fs_hz:.3f} seconds)")
    p_a = np.mean(np.abs(sa)**2)
    p_b = np.mean(np.abs(sb)**2)
    print(f"   Channel A Power: {p_a:.2e} | Channel B Power: {p_b:.2e}")

    # 2. Beamform SUM channel
    print(f"2. Performing Phased Array Beamforming (Phase Offset: {phase_offset_deg:.1f}°)...")
    s_sum = pr.beamform_sum(sa, sb, phase_offset_deg)
    p_sum = np.mean(np.abs(s_sum)**2)
    print(f"   Sum Channel Power: {p_sum:.2e}")

    # Select Ref and Sur
    s_ref, s_sur = pr.select_channels(sa, sb, s_sum, ref_source, sur_source)

    # Plot channel spectrum
    spec_plot_path = os.path.join(output_dir, f"{prefix}channels_spectrum.png")
    pr.plot_channels_and_spectra(sa, sb, s_sum, fs_hz,
                                 title=f"Channel Signals & Spectra ({os.path.basename(file_a)})",
                                 save_path=spec_plot_path, show=show_plots)
    print(f"   Saved spectrum plot -> {spec_plot_path}")

    # 3. Autocorrelation analysis
    print("3. Computing Reference Signal Autocorrelation...")
    lags, delays_us, equiv_range_km, r_norm = pr.compute_autocorrelation(s_ref, fs_hz, max_lags=500)
    autocorr_plot_path = os.path.join(output_dir, f"{prefix}autocorrelation.png")
    pr.plot_autocorrelation(lags, delays_us, equiv_range_km, r_norm,
                            title=f"Ref Channel Autocorrelation ({ref_source} Channel)",
                            save_path=autocorr_plot_path, show=show_plots)
    print(f"   Saved autocorrelation plot -> {autocorr_plot_path}")

    # 4. ECA-B Clutter Cancellation
    print(f"4. Applying ECA-B Clutter Cancellation ({clutter_taps} taps)...")
    s_sur_clean = pr.eca_clutter_cancellation(s_ref, s_sur, max_delay_taps=clutter_taps)
    att_db = 10 * np.log10(np.mean(np.abs(s_sur)**2) / (np.mean(np.abs(s_sur_clean)**2) + 1e-12))
    print(f"   Clutter/DPI Attenuation achieved: {att_db:.2f} dB")

    # 5. Cross-Ambiguity Function (CAF) / Range-Doppler Map (RDM)
    print("5. Computing Cross-Ambiguity Function (Range-Doppler Map)...")
    rdm_db, range_km, doppler_hz, velocity_kmh = pr.compute_cross_ambiguity_function(
        s_ref, s_sur_clean, fs_hz, fc_hz,
        max_range_km=max_range_km,
        max_doppler_hz=max_doppler_hz,
        doppler_bins=512
    )
    print(f"   RDM Matrix Shape: {rdm_db.shape} (Range bins: {len(range_km)}, Doppler bins: {len(doppler_hz)})")

    # 6. CFAR Target Detection
    print(f"6. Running 2D CFAR Detector (Threshold: {cfar_thresh_db} dB above noise floor)...")
    detections = pr.cfar_2d(
        rdm_db,
        guard_cells=(2, 4),
        ref_cells=(8, 16),
        threshold_db=cfar_thresh_db,
        exclude_zero_doppler_hz=15.0,
        doppler_axis=doppler_hz,
        range_axis=range_km,
        velocity_axis=velocity_kmh
    )

    print("-" * 50)
    print(f"AIRPLANE TARGET DETECTION SUMMARY ({len(detections)} candidate targets detected):")
    if detections:
        print(f"{'Target #':<10}{'Range (km)':<15}{'Velocity (km/h)':<18}{'Doppler (Hz)':<15}{'SNR (dB)':<10}")
        print("-" * 68)
        for idx, det in enumerate(detections, 1):
            print(f"{idx:<10}{det['range_km']:<15.2f}{det['velocity_kmh']:<18.1f}{det['doppler_hz']:<15.1f}{det['snr_db']:<10.2f}")
    else:
        print("   No aircraft reflections detected above CFAR threshold in this block.")
    print("-" * 50)

    # Plot RDM
    rdm_plot_path = os.path.join(output_dir, f"{prefix}range_doppler_map.png")
    pr.plot_range_doppler_map(
        rdm_db, range_km, velocity_kmh, doppler_hz,
        detections=detections,
        title=f"Range-Doppler Map ({os.path.basename(file_a)} & {os.path.basename(file_b)}) - Fc {fc_hz/1e6:.1f} MHz",
        save_path=rdm_plot_path, show=show_plots
    )
    print(f"   Saved Range-Doppler Map plot -> {rdm_plot_path}")
    print("=" * 70 + "\n")

    return detections


def search_all_recordings(iq_dir="IQ", fs_default=640000.0, output_dir="output_plots"):
    """
    Automatically locate paired channel A and B files in IQ directory and scan for airplanes.
    """
    if not os.path.exists(iq_dir):
        print(f"Directory {iq_dir} not found!")
        return

    # Find paired files starting with 'a' and 'b'
    all_files = os.listdir(iq_dir)
    a_files = sorted([f for f in all_files if f.startswith("a") and not f.endswith(".sigmf")])
    
    pairs = []
    for fa in a_files:
        # Corresponding b file: replace leading 'a' with 'b'
        # Handle cases like 'a102_640' -> 'b102_640', 'aa102_640' -> 'bb102_640', 'aaa102_640' -> 'bab102_640' or 'bbb102_640'
        candidate_b = 'b' + fa[1:]
        if candidate_b in all_files:
            pairs.append((fa, candidate_b))
        elif fa == "aaa102_640" and "bab102_640" in all_files:
            pairs.append((fa, "bab102_640"))

    print(f"Found {len(pairs)} coherent dual-channel file pairs in '{iq_dir}':")
    for fa, fb in pairs:
        print(f"  - ({fa}, {fb})")
    print("\nStarting automated airplane search across datasets...\n")

    all_detections = {}

    for fa, fb in pairs:
        file_a = os.path.join(iq_dir, fa)
        file_b = os.path.join(iq_dir, fb)

        # Infer Fc from filename if possible
        if "951" in fa:
            fc = 95.1e6
        elif "1055" in fa or "105" in fa:
            fc = 105.5e6
        elif "1065" in fa:
            fc = 106.5e6
        else:
            fc = 102.0e6  # Default FM carrier

        # Infer Fs
        if "3m" in fa:
            fs = 3.0e6
        else:
            fs = fs_default

        prefix = f"{fa}_"
        dets = analyze_dataset(
            file_a, file_b, fs_hz=fs, fc_hz=fc,
            phase_offset_deg=0.0, cpi_sec=1.0,
            output_dir=output_dir, prefix=prefix, show_plots=False
        )
        all_detections[fa] = dets

    print("=" * 70)
    print("AUTOMATED SEARCH COMPLETE ACROSS ALL IQ RECORDINGS:")
    total_dets = 0
    for name, dets in all_detections.items():
        print(f"  {name}: {len(dets)} target candidate(s) detected")
        total_dets += len(dets)
    print(f"TOTAL AIRPLANE CANDIDATES DETECTED: {total_dets}")
    print("=" * 70)


def main():
    args = parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    pa_cfg = cfg.get("phased_array", {})
    proc_cfg = cfg.get("processing", {})
    cfar_cfg = cfg.get("cfar", {})
    vis_cfg = cfg.get("visualization", {})

    file_a = args.file_a or os.path.join(ds_cfg.get("dir", "IQ"), ds_cfg.get("channel_a_file", "a102_640"))
    file_b = args.file_b or os.path.join(ds_cfg.get("dir", "IQ"), ds_cfg.get("channel_b_file", "b102_640"))
    fs_hz = args.fs or ds_cfg.get("fs_hz", 640000.0)
    fc_hz = args.fc or ds_cfg.get("fc_hz", 102000000.0)
    phase_offset_deg = args.phase if args.phase is not None else pa_cfg.get("phase_offset_deg", 0.0)
    cpi_sec = args.cpi or ds_cfg.get("cpi_seconds", 1.0)
    show_plots = args.show or vis_cfg.get("show_plots", False)

    if args.search_all:
        search_all_recordings(iq_dir=ds_cfg.get("dir", "IQ"), fs_default=fs_hz, output_dir=vis_cfg.get("output_dir", "output_plots"))
    else:
        analyze_dataset(
            file_a=file_a,
            file_b=file_b,
            fs_hz=fs_hz,
            fc_hz=fc_hz,
            phase_offset_deg=phase_offset_deg,
            cpi_sec=cpi_sec,
            offset_samples=ds_cfg.get("offset_samples", 0),
            ref_source=pa_cfg.get("reference_source", "SUM"),
            sur_source=pa_cfg.get("surveillance_source", "B"),
            max_range_km=proc_cfg.get("max_range_km", 50.0),
            max_doppler_hz=proc_cfg.get("max_doppler_hz", 500.0),
            clutter_taps=proc_cfg.get("clutter_cancellation", {}).get("filter_taps", 40),
            cfar_thresh_db=cfar_cfg.get("snr_threshold_db", 12.0),
            output_dir=vis_cfg.get("output_dir", "output_plots"),
            prefix="",
            show_plots=show_plots
        )


if __name__ == "__main__":
    main()
