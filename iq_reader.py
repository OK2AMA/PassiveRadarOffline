"""
IQ Reader module for 16I16Q format captured by AD9363 (Tezuka FW).

Format details:
- 16-bit signed integer I (little endian, int16)
- 16-bit signed integer Q (little endian, int16)
- 4 bytes per complex sample
"""

import os
import numpy as np


class IQReader:
    """Class to load and inspect 16I16Q dual-channel binary files."""

    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IQ file not found: {filepath}")
        self.file_size_bytes = os.path.getsize(filepath)
        self.total_samples = self.file_size_bytes // 4

    def read_samples(self, num_samples=None, offset_samples=0, remove_dc=True, normalize=True):
        """
        Read complex samples from file.

        Parameters:
        -----------
        num_samples : int or None
            Number of complex samples to read. Reads all if None.
        offset_samples : int
            Sample offset from start of file.
        remove_dc : bool
            If True, subtract mean (DC component).
        normalize : bool
            If True, scale values to [-1.0, 1.0].

        Returns:
        --------
        np.ndarray (complex64):
            Array of complex IQ samples.
        """
        count = num_samples * 2 if num_samples is not None else -1
        offset_bytes = offset_samples * 4

        raw = np.fromfile(self.filepath, dtype=np.int16, count=count, offset=offset_bytes)
        
        # De-interleave I and Q
        i_data = raw[0::2].astype(np.float32)
        q_data = raw[1::2].astype(np.float32)

        if remove_dc:
            i_data -= np.mean(i_data)
            q_data -= np.mean(q_data)

        if normalize:
            # 16-bit signed integer max is 32768
            scale = 32768.0
            i_data /= scale
            q_data /= scale

        return i_data + 1j * q_data

    def get_duration_seconds(self, fs_hz):
        """Return file duration in seconds given sample rate."""
        return self.total_samples / float(fs_hz)


def load_coherent_pair(file_a, file_b, num_samples=None, offset_samples=0, remove_dc=True, normalize=True):
    """
    Load coherent dual-channel sample pair (channel A and channel B).

    Returns:
    --------
    sa, sb : np.ndarray (complex64)
        Complex IQ sample arrays for channel A and B.
    """
    reader_a = IQReader(file_a)
    reader_b = IQReader(file_b)

    sa = reader_a.read_samples(num_samples, offset_samples, remove_dc, normalize)
    sb = reader_b.read_samples(num_samples, offset_samples, remove_dc, normalize)

    # Ensure equal length
    min_len = min(len(sa), len(sb))
    return sa[:min_len], sb[:min_len]
