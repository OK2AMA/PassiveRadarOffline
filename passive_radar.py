"""
Passive Radar Processing Engine.

Provides signal processing routines for:
- Phased array spatial beamforming (channel sum with phase shift)
- Autocorrelation analysis of reference signal
- Direct Path Interference (DPI) & ECA-B clutter cancellation
- Cross-Ambiguity Function (CAF) / Range-Doppler Map (RDM) generation
- 2D CFAR target detection
- Visualization suite for spectrum, autocorrelation, and RDM heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt

C_LIGHT = 299792458.0  # Speed of light in m/s


def beamform_sum(sa, sb, phase_offset_deg):
    """
    Combine Channel A and Channel B with phase offset (Phased Array Beamforming).

    S_sum(t) = S_a(t) + S_b(t) * exp(j * phase_offset_rad)

    Parameters:
    -----------
    sa, sb : np.ndarray
        Complex IQ sample streams for channel A and B.
    phase_offset_deg : float
        Phase offset in degrees applied to channel B relative to A.

    Returns:
    --------
    s_sum : np.ndarray
        Summed complex IQ signal.
    """
    phase_rad = np.radians(phase_offset_deg)
    phase_factor = np.exp(1j * phase_rad)
    return sa + sb * phase_factor


def select_channels(sa, sb, s_sum, ref_source="SUM", sur_source="B"):
    """Select reference and surveillance signals based on config source string."""
    sources = {
        "A": sa,
        "B": sb,
        "SUM": s_sum
    }

    ref_sig = sources.get(ref_source.upper(), s_sum)
    sur_sig = sources.get(sur_source.upper(), sb)

    return ref_sig, sur_sig


def compute_autocorrelation(signal_iq, fs_hz, max_lags=500):
    """
    Compute normalized autocorrelation of a complex signal.

    Parameters:
    -----------
    signal_iq : np.ndarray
        Complex IQ signal.
    fs_hz : float
        Sample rate in Hz.
    max_lags : int
        Maximum lag count in samples.

    Returns:
    --------
    lags : np.ndarray
        Lag vector in samples.
    delays_us : np.ndarray
        Delay vector in microseconds.
    equiv_range_km : np.ndarray
        Equivalent bistatic range spacing in kilometers (c * tau).
    r_norm : np.ndarray
        Normalized autocorrelation values (dB scale and linear amplitude).
    """
    N = len(signal_iq)
    # Zero-pad and compute FFT-based correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.fft(signal_iq, n_fft)
    corr = np.fft.ifft(X * np.conj(X))
    
    # Extract non-negative lags up to max_lags
    r = np.abs(corr[:max_lags])
    # Normalize by zero-lag (power)
    r_norm = r / r[0]

    lags = np.arange(max_lags)
    delays_us = (lags / float(fs_hz)) * 1e6
    equiv_range_km = (lags * C_LIGHT / float(fs_hz)) / 1000.0

    return lags, delays_us, equiv_range_km, r_norm


def eca_clutter_cancellation(s_ref, s_sur, max_delay_taps=40):
    """
    Extensive Cancellation Algorithm - Batch (ECA-B) for clutter cancellation.

    Projects surveillance signal onto subspace spanned by reference signal and its
    delayed versions, removing direct path interference and static ground clutter.

    Parameters:
    -----------
    s_ref, s_sur : np.ndarray
        Reference and Surveillance complex IQ sample streams.
    max_delay_taps : int
        Number of delay taps (range clutter span) to cancel.

    Returns:
    --------
    s_clean : np.ndarray
        Surveillance signal with clutter subtracted.
    """
    N = len(s_ref)
    if N <= max_delay_taps:
        return s_sur

    # Construct delay matrix X of size (N - max_delay_taps, max_delay_taps + 1)
    X = np.zeros((N - max_delay_taps, max_delay_taps + 1), dtype=np.complex64)
    for tap in range(max_delay_taps + 1):
        X[:, tap] = s_ref[max_delay_taps - tap : N - tap]

    y = s_sur[max_delay_taps:]

    # Least squares fit X * w ~ y
    weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_clean = y - X @ weights

    # Zero-pad initial samples
    s_clean = np.zeros(N, dtype=np.complex64)
    s_clean[max_delay_taps:] = y_clean
    return s_clean


def compute_cross_ambiguity_function(s_ref, s_sur, fs_hz, fc_hz, max_range_km=50.0, max_doppler_hz=500.0, doppler_bins=None):
    """
    Compute Cross-Ambiguity Function (CAF) / Range-Doppler Map (RDM).
    Optimized strided matrix execution for maximum calculation speed.

    Parameters:
    -----------
    s_ref, s_sur : np.ndarray
        Cleaned complex reference and surveillance signals.
    fs_hz : float
        Sampling frequency in Hz.
    fc_hz : float
        Center frequency in Hz.
    max_range_km : float
        Maximum range limit in km.
    max_doppler_hz : float
        Maximum Doppler frequency limit in Hz.

    Returns:
    --------
    rdm_db : np.ndarray (shape: max_range_bins x num_doppler_samples)
        Range-Doppler map in dB normalized to peak.
    range_axis_km : np.ndarray
        Range axis in kilometers.
    doppler_axis_hz : np.ndarray
        Doppler frequency axis in Hz.
    velocity_axis_kmh : np.ndarray
        Bistatic velocity axis in km/h.
    """
    N = len(s_ref)
    range_resolution_m = C_LIGHT / fs_hz
    max_range_bins = int(np.ceil((max_range_km * 1000.0) / range_resolution_m))
    max_range_bins = max(1, max_range_bins)

    target_fs = 2.0 * max_doppler_hz
    decimation = max(1, int(fs_hz / target_fs))
    fs_eff = fs_hz / decimation

    num_doppler_samples = N // decimation
    L = num_doppler_samples * decimation

    # Zero-pad reference signal for strided range shifting
    pad_ref = np.pad(s_ref[:L], (max_range_bins, 0), mode='constant')

    # Construct strided matrix of shape (max_range_bins, L)
    shape = (max_range_bins, L)
    ref_matrix = np.lib.stride_tricks.as_strided(
        pad_ref[max_range_bins:],
        shape=shape,
        strides=(-pad_ref.strides[0], pad_ref.strides[0])
    )

    # Reshape matrices for vectorized block downsampling
    ref_blocks = ref_matrix.reshape(max_range_bins, num_doppler_samples, decimation)
    sur_blocks = s_sur[:L].reshape(1, num_doppler_samples, decimation)

    # Vectorized block mean of cross-product
    dec_prod = np.mean(sur_blocks * np.conj(ref_blocks), axis=2)

    win = np.hanning(num_doppler_samples)[None, :]
    fft_res = np.fft.fftshift(np.fft.fft(dec_prod * win, axis=1), axes=1)
    rdm = np.abs(fft_res)

    rdm_db = 20.0 * np.log10(rdm + 1e-12)
    rdm_db -= np.max(rdm_db)  # Normalize peak to 0 dB

    range_axis_km = (np.arange(max_range_bins) * range_resolution_m) / 1000.0
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(num_doppler_samples, d=1.0 / fs_eff))
    velocity_axis_kmh = (doppler_axis_hz * C_LIGHT / fc_hz) * 3.6

    return rdm_db, range_axis_km, doppler_axis_hz, velocity_axis_kmh


def suppress_zero_doppler(rdm_db, doppler_axis_hz, notch_bw_hz=20.0, notch_depth_db=-80.0):
    """
    Apply Zero-Doppler notch filter to Range-Doppler Map matrix.
    Suppresses residual static ground clutter and direct path transmitter leakage around 0 Hz.

    Parameters:
    -----------
    rdm_db : np.ndarray
        2D Range-Doppler Map in dB.
    doppler_axis_hz : np.ndarray
        Doppler frequency axis in Hz.
    notch_bw_hz : float
        Notch bandwidth in Hz (+/- notch_bw_hz around 0 Hz).
    notch_depth_db : float
        dB value to assign to notched zero-Doppler region.

    Returns:
    --------
    rdm_notched_db : np.ndarray
        Range-Doppler Map with 0 Hz Doppler region suppressed.
    """
    rdm_notched_db = np.copy(rdm_db)
    zero_doppler_mask = np.abs(doppler_axis_hz) <= notch_bw_hz
    rdm_notched_db[:, zero_doppler_mask] = notch_depth_db
    return rdm_notched_db


def cfar_2d(rdm_db, guard_cells=(2, 4), ref_cells=(8, 16), threshold_db=12.0, exclude_zero_doppler_hz=20.0, doppler_axis=None, range_axis=None, velocity_axis=None):
    """
    2D Cell-Averaging CFAR detector on Range-Doppler Map.

    Parameters:
    -----------
    rdm_db : np.ndarray
        2D Range-Doppler Map matrix in dB.
    guard_cells : tuple (range_guard, doppler_guard)
        Number of guard cells in range and doppler dimensions.
    ref_cells : tuple (range_ref, doppler_ref)
        Number of reference cells in range and doppler dimensions.
    threshold_db : float
        SNR threshold above estimated noise background in dB.
    exclude_zero_doppler_hz : float
        Doppler region around 0 Hz to exclude from target search.

    Returns:
    --------
    detections : list of dict
        Detected targets with range, doppler, velocity, and SNR.
    """
    num_range, num_doppler = rdm_db.shape
    gr, gd = guard_cells
    rr, rd = ref_cells

    detections = []

    # Calculate zero doppler indices to exclude
    if doppler_axis is not None and exclude_zero_doppler_hz > 0:
        zero_doppler_mask = np.abs(doppler_axis) <= exclude_zero_doppler_hz
    else:
        zero_doppler_mask = np.zeros(num_doppler, dtype=bool)

    # Convert dB to linear power for CFAR averaging
    power_lin = 10.0 ** (rdm_db / 10.0)

    for r in range(rr + gr, num_range - (rr + gr)):
        for d in range(rd + gd, num_doppler - (rd + gd)):
            if zero_doppler_mask[d]:
                continue

            # Extract window around CUT (Cell Under Test)
            win = power_lin[r - (rr + gr) : r + (rr + gr) + 1, d - (rd + gd) : d + (rd + gd) + 1]
            
            # Mask out CUT and guard cells
            mask = np.ones_like(win, dtype=bool)
            mask[rr : rr + 2 * gr + 1, rd : rd + 2 * gd + 1] = False

            # Noise floor calculation
            ref_values = win[mask]
            noise_floor_lin = np.mean(ref_values)
            noise_floor_db = 10.0 * np.log10(noise_floor_lin + 1e-12)

            cut_db = rdm_db[r, d]
            snr_db = cut_db - noise_floor_db

            if snr_db >= threshold_db:
                r_km = range_axis[r] if range_axis is not None else float(r)
                d_hz = doppler_axis[d] if doppler_axis is not None else float(d)
                v_kmh = velocity_axis[d] if velocity_axis is not None else 0.0

                detections.append({
                    "range_bin": r,
                    "doppler_bin": d,
                    "range_km": float(r_km),
                    "doppler_hz": float(d_hz),
                    "velocity_kmh": float(v_kmh),
                    "snr_db": float(snr_db),
                    "power_db": float(cut_db)
                })

    return detections


def plot_channels_and_spectra(sa, sb, s_sum, fs_hz, title="Channel Signals and Spectra", save_path=None, show=False):
    """Plot time domain power and FFT spectrum of Ch A, Ch B, and Sum channel."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    t = np.arange(min(len(sa), 5000)) / fs_hz * 1e3  # ms

    # Time domain magnitude
    axs[0, 0].plot(t, np.abs(sa[:len(t)]), label="Ch A", alpha=0.8, color="#1f77b4")
    axs[0, 0].plot(t, np.abs(sb[:len(t)]), label="Ch B", alpha=0.8, color="#ff7f0e")
    axs[0, 0].plot(t, np.abs(s_sum[:len(t)]), label="Sum", alpha=0.8, color="#2ca02c")
    axs[0, 0].set_title("Time Domain Magnitude (First 5000 samples)")
    axs[0, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Magnitude")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()

    # Phase difference A vs B
    phase_diff_deg = np.degrees(np.angle(sa[:len(t)] * np.conj(sb[:len(t)])))
    axs[0, 1].plot(t, phase_diff_deg, color="#9467bd", alpha=0.7)
    axs[0, 1].set_title("Phase Difference Ch A vs Ch B")
    axs[0, 1].set_xlabel("Time (ms)")
    axs[0, 1].set_ylabel("Phase Diff (deg)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # FFT Spectrum
    n_fft = 4096
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs_hz)) / 1e3  # kHz
    
    spec_a = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sa[:n_fft] * np.hanning(n_fft)))) + 1e-6)
    spec_b = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sb[:n_fft] * np.hanning(n_fft)))) + 1e-6)
    spec_sum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(s_sum[:n_fft] * np.hanning(n_fft)))) + 1e-6)

    axs[1, 0].plot(freqs, spec_a, label="Ch A", color="#1f77b4", alpha=0.8)
    axs[1, 0].plot(freqs, spec_b, label="Ch B", color="#ff7f0e", alpha=0.8)
    axs[1, 0].plot(freqs, spec_sum, label="Sum", color="#2ca02c", alpha=0.8)
    axs[1, 0].set_title("Spectrum (Baseband)")
    axs[1, 0].set_xlabel("Frequency (kHz)")
    axs[1, 0].set_ylabel("Power (dB)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()

    # IQ Scatter plot
    axs[1, 1].scatter(sa[:1000].real, sa[:1000].imag, s=5, alpha=0.5, label="Ch A", color="#1f77b4")
    axs[1, 1].scatter(sb[:1000].real, sb[:1000].imag, s=5, alpha=0.5, label="Ch B", color="#ff7f0e")
    axs[1, 1].set_title("IQ Constellation Diagram")
    axs[1, 1].set_xlabel("In-Phase (I)")
    axs[1, 1].set_ylabel("Quadrature (Q)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_autocorrelation(lags, delays_us, equiv_range_km, r_norm, title="Reference Signal Autocorrelation", save_path=None, show=False):
    """Plot normalized autocorrelation of reference signal."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    r_db = 20 * np.log10(r_norm + 1e-6)

    ax1.plot(equiv_range_km, r_db, color="#0072BD", linewidth=1.5, label="Autocorrelation (dB)")
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.set_xlabel("Equivalent Bistatic Range Lag (km)")
    ax1.set_ylabel("Normalized Autocorrelation (dB)", color="#0072BD")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim([-40, 2])

    ax2 = ax1.twiny()
    ax2.set_xlim([delays_us[0], delays_us[-1]])
    ax2.set_xlabel("Delay Lag (μs)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_annotated_range_doppler_map(rdm_db, range_axis_km, velocity_axis_kmh, doppler_axis_hz,
                                    metadata=None, detections=None, title="Range-Doppler Map",
                                    save_path=None, show=False):
    """
    Plot Range-Doppler Map heatmap with rich technical annotations, phase offsets,
    power levels, system specs, and target markers.
    """
    fig, ax = plt.subplots(figsize=(13, 8))

    vmin = max(np.min(rdm_db), -60.0)
    vmax = 0.0

    im = ax.imshow(
        rdm_db,
        aspect='auto',
        origin='lower',
        extent=[velocity_axis_kmh[0], velocity_axis_kmh[-1], range_axis_km[0], range_axis_km[-1]],
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Relative Power (dB)", fontsize=11, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Bistatic Velocity (km/h)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Bistatic Range (km)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.35, color="white")

    # Zero-Doppler Line Marker
    ax.axvline(0, color='cyan', linestyle=':', linewidth=1.5, alpha=0.7, label="Zero Doppler (0 km/h)")

    # 1. Top-Left System Metadata Box
    if metadata:
        meta_text = (
            f"SYSTEM METADATA:\n"
            f" Carrier Fc:  {metadata.get('fc_mhz', 0.0):.2f} MHz\n"
            f" Sample Rate: {metadata.get('fs_mhz', 0.0):.2f} MHz\n"
            f" Range Res:   {metadata.get('range_res_m', 0.0):.1f} m/bin\n"
            f" CPI Time:    {metadata.get('cpi_sec', 0.0):.1f} sec\n"
            f" Doppler Res: {metadata.get('doppler_res_hz', 0.0):.2f} Hz"
        )
        ax.text(
            0.02, 0.96, meta_text, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace', fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="#111111", ec="#00FF00", alpha=0.85),
            color="#00FF00"
        )

        # 2. Top-Right Phased Array & Power Measurements Box
        pa_text = (
            f"PHASE & POWER MEASUREMENTS:\n"
            f" Phase Offset (Δφ): {metadata.get('phase_deg', 0.0):+.2f}° ({metadata.get('phase_rad', 0.0):+.3f} rad)\n"
            f" Coherence (γ12):   {metadata.get('coherence', 0.0):.4f}\n"
            f" Ch1 Power (P1):   {metadata.get('p_ch1_db', 0.0):.2f} dB\n"
            f" Ch2 Power (P2):   {metadata.get('p_ch2_db', 0.0):.2f} dB\n"
            f" S_max Direct Gain: {metadata.get('direct_gain_db', 0.0):+.2f} dB\n"
            f" S_null Null Depth: {metadata.get('null_depth_db', 0.0):.2f} dB"
        )
        ax.text(
            0.98, 0.96, pa_text, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace', fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", fc="#111111", ec="#FFD700", alpha=0.85),
            color="#FFD700"
        )

    # 3. Target Detections Overlay
    if detections:
        det_v = [d["velocity_kmh"] for d in detections]
        det_r = [d["range_km"] for d in detections]
        ax.scatter(det_v, det_r, color="red", marker="o", s=90, facecolors='none', edgecolors='red', linewidth=2, label=f"Detections ({len(detections)})")
        for idx, d in enumerate(detections, 1):
            ax.annotate(
                f"T{idx}: {d['snr_db']:.1f} dB\n({d['velocity_kmh']:+.0f} km/h, {d['range_km']:.1f} km)",
                (d["velocity_kmh"], d["range_km"]),
                textcoords="offset points",
                xytext=(10, 10),
                ha='left',
                color='#00FFFF',
                fontsize=8.5,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.75, ec="#00FFFF")
            )

    ax.legend(loc="lower right", facecolor="#111111", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close()


def plot_range_doppler_map(rdm_db, range_axis_km, velocity_axis_kmh, doppler_axis_hz, detections=None, title="Range-Doppler Map", save_path=None, show=False):
    """Plot Range-Doppler Map heatmap with detections."""
    fig, ax = plt.subplots(figsize=(12, 7))

    vmin = max(np.min(rdm_db), -60.0)
    vmax = 0.0

    im = ax.imshow(
        rdm_db,
        aspect='auto',
        origin='lower',
        extent=[velocity_axis_kmh[0], velocity_axis_kmh[-1], range_axis_km[0], range_axis_km[-1]],
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative Power (dB)")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Bistatic Velocity (km/h)")
    ax.set_ylabel("Bistatic Range (km)")
    ax.grid(True, linestyle="--", alpha=0.4, color="white")

    # Plot detected airplane targets
    if detections:
        det_v = [d["velocity_kmh"] for d in detections]
        det_r = [d["range_km"] for d in detections]
        snrs = [d["snr_db"] for d in detections]

        ax.scatter(det_v, det_r, color="red", marker="o", s=80, facecolors='none', edgecolors='red', linewidth=2, label=f"Detections ({len(detections)})")
        for d in detections:
            ax.annotate(
                f"{d['snr_db']:.1f} dB\n({d['velocity_kmh']:.0f} km/h, {d['range_km']:.1f} km)",
                (d["velocity_kmh"], d["range_km"]),
                textcoords="offset points",
                xytext=(10, 10),
                ha='left',
                color='yellow',
                fontsize=9,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="yellow")
            )

        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()
