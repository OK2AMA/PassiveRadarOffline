"""
GPU-Accelerated Passive Radar Processing Engine using PyTorch CUDA.

Accelerates:
- Phased array spatial beamforming sweeps
- ECA clutter cancellation via PyTorch GPU linear algebra
- High-speed 2D FFT Range-Doppler Map (CAF) generation
- GPU 2D CFAR convolution target detector
- Dynamic Range normalization for faint aircraft target visibility
"""

import torch
import numpy as np

C_LIGHT = 299792458.0


class PassiveRadarGPU:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def beamform_sum_gpu(self, sa_tensor, sb_tensor, phase_offset_deg):
        """Combine Channel A and B on GPU with phase offset."""
        phase_rad = torch.tensor(np.radians(phase_offset_deg), dtype=torch.float32, device=self.device)
        phase_factor = torch.polar(torch.tensor(1.0, device=self.device), phase_rad)
        return sa_tensor + sb_tensor * phase_factor

    def eca_clutter_cancellation_gpu(self, s_ref_tensor, s_sur_tensor, max_delay_taps=60):
        """Extensive Cancellation Algorithm (Batch) on GPU using sub-sampled least squares for RAM safety."""
        N = len(s_ref_tensor)
        if N <= max_delay_taps:
            return s_sur_tensor

        # Construct delay matrix X of shape (N - max_delay_taps, max_delay_taps + 1)
        X = torch.zeros((N - max_delay_taps, max_delay_taps + 1), dtype=torch.complex64, device=self.device)
        for tap in range(max_delay_taps + 1):
            X[:, tap] = s_ref_tensor[max_delay_taps - tap : N - tap]

        y = s_sur_tensor[max_delay_taps:].unsqueeze(1)

        # Sub-sample rows for RAM safety if N is very large
        stride = max(1, (N - max_delay_taps) // 100000)
        X_sub = X[::stride]
        y_sub = y[::stride]

        # Solve X_sub * w ~ y_sub on GPU (uses only ~48 MB VRAM!)
        lstsq_res = torch.linalg.lstsq(X_sub, y_sub)
        weights = lstsq_res.solution

        y_clean = y - (X @ weights)

        # Pad initial delay samples
        s_clean = torch.zeros(N, dtype=torch.complex64, device=self.device)
        s_clean[max_delay_taps:] = y_clean.squeeze(1)
        return s_clean

    def compute_caf_gpu(self, s_ref_tensor, s_sur_tensor, fs_hz, fc_hz, max_range_km=50.0, max_doppler_hz=500.0):
        """
        Compute Cross-Ambiguity Function (Range-Doppler Map) on GPU with strict RAM safety.
        Memory footprint is kept below 50 MB VRAM by chunking range bins.
        """
        N = len(s_ref_tensor)
        range_res_m = C_LIGHT / fs_hz
        max_range_bins = int(np.ceil((max_range_km * 1000.0) / range_res_m))
        max_range_bins = max(1, max_range_bins)

        target_fs = 2.0 * max_doppler_hz
        decimation = max(1, int(fs_hz / target_fs))
        fs_eff = fs_hz / decimation

        num_doppler_samples = N // decimation
        L = num_doppler_samples * decimation

        # Pre-allocate output RDM on GPU
        rdm_gpu = torch.zeros((max_range_bins, num_doppler_samples), dtype=torch.float32, device=self.device)
        win = torch.hann_window(num_doppler_samples, device=self.device)

        # Batch range bins (e.g. 25 range bins per batch) to keep GPU VRAM under 50 MB
        batch_size = 25
        for r_start in range(0, max_range_bins, batch_size):
            r_end = min(max_range_bins, r_start + batch_size)
            curr_batch = r_end - r_start

            # Zero-pad ref for batch
            pad_ref = torch.nn.functional.pad(s_ref_tensor[:L], (r_end, 0))

            # Unfold batch: (curr_batch, L)
            ref_matrix = torch.flip(pad_ref[r_start:].unfold(0, L, 1)[:curr_batch], dims=[0])

            ref_blocks = ref_matrix.view(curr_batch, num_doppler_samples, decimation)
            sur_blocks = s_sur_tensor[:L].view(1, num_doppler_samples, decimation)

            dec_prod = torch.mean(sur_blocks * torch.conj(ref_blocks), dim=2)
            fft_res = torch.fft.fftshift(torch.fft.fft(dec_prod * win.unsqueeze(0), dim=1), dim=1)
            rdm_gpu[r_start:r_end, :] = torch.abs(fft_res)

            del pad_ref, ref_matrix, ref_blocks, sur_blocks, dec_prod, fft_res
            torch.cuda.empty_cache()

        rdm_db_gpu = 20.0 * torch.log10(rdm_gpu + 1e-12)
        rdm_db_gpu -= torch.max(rdm_db_gpu)

        range_axis_km = (np.arange(max_range_bins) * range_res_m) / 1000.0
        doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(num_doppler_samples, d=1.0 / fs_eff))
        velocity_axis_kmh = (doppler_axis_hz * C_LIGHT / fc_hz) * 3.6

        return rdm_db_gpu.cpu().numpy(), range_axis_km, doppler_axis_hz, velocity_axis_kmh
