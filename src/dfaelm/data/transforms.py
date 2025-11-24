"""
Data transformation utilities for EEG preprocessing
"""

import torch
import numpy as np
from scipy import signal
import pywt


class EEGTransform:
    """Base class for EEG transformations"""
    def __call__(self, data):
        raise NotImplementedError


class ZScoreNormalize(EEGTransform):
    """Z-score normalization"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if self.mean is None or self.std is None:
            # Compute per-channel statistics
            self.mean = data.mean(dim=-1, keepdim=True)
            self.std = data.std(dim=-1, keepdim=True)

        return (data - self.mean) / (self.std + 1e-8)


class BandpassFilter(EEGTransform):
    """Bandpass filter for EEG signals"""
    def __init__(self, low_freq=1.0, high_freq=30.0, fs=500, order=4):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order

        # Design Butterworth filter
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        self.b, self.a = signal.butter(order, [low, high], btype='band')

    def __call__(self, data):
        # Apply filter along time dimension
        filtered = signal.filtfilt(self.b, self.a, data.numpy(), axis=-1)
        return torch.from_numpy(filtered).float()


class NotchFilter(EEGTransform):
    """Notch filter for powerline noise removal"""
    def __init__(self, freq=50.0, fs=500, quality_factor=30):
        self.freq = freq
        self.fs = fs
        self.quality_factor = quality_factor

        # Design notch filter
        nyquist = fs / 2
        notch_freq = freq / nyquist
        self.b, self.a = signal.iirnotch(notch_freq, quality_factor)

    def __call__(self, data):
        # Apply filter along time dimension
        filtered = signal.filtfilt(self.b, self.a, data.numpy(), axis=-1)
        return torch.from_numpy(filtered).float()


class AverageReference(EEGTransform):
    """Apply average reference montage"""
    def __call__(self, data):
        # Compute average across channels
        avg_ref = data.mean(dim=1, keepdim=True)
        # Subtract average reference
        return data - avg_ref


class RandomCrop(EEGTransform):
    """Random crop of signal to specified length"""
    def __init__(self, length):
        self.length = length

    def __call__(self, data):
        seq_len = data.shape[-1]
        if seq_len <= self.length:
            return data

        start = np.random.randint(0, seq_len - self.length)
        return data[..., start:start + self.length]


class Windowing(EEGTransform):
    """Create overlapping windows from continuous signal"""
    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step = int(window_size * (1 - overlap))

    def __call__(self, data):
        # data shape: [channels, time]
        channels, time_len = data.shape

        windows = []
        for start in range(0, time_len - self.window_size + 1, self.step):
            end = start + self.window_size
            window = data[:, start:end]
            windows.append(window)

        # Stack windows: [num_windows, channels, window_size]
        if windows:
            return torch.stack(windows, dim=0)
        else:
            return torch.empty(0, channels, self.window_size)


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class EEGPreprocessingPipeline:
    """Complete EEG preprocessing pipeline"""
    def __init__(self, fs=500, window_size=5.0, overlap=0.5):
        """
        Args:
            fs (float): Sampling frequency
            window_size (float): Window size in seconds
            overlap (float): Overlap fraction
        """
        window_samples = int(window_size * fs)

        self.pipeline = Compose([
            NotchFilter(freq=50.0, fs=fs),  # Remove powerline noise
            BandpassFilter(low_freq=1.0, high_freq=30.0, fs=fs),  # EEG band
            AverageReference(),  # Average reference montage
            ZScoreNormalize(),  # Z-score normalization
            Windowing(window_samples, overlap)  # Create windows
        ])

    def __call__(self, data):
        return self.pipeline(data)


class FeatureExtractor:
    """Extract time and frequency domain features from EEG windows"""

    def __init__(self, fs=500):
        self.fs = fs

    def extract_time_features(self, window):
        """Extract time-domain features"""
        features = []

        # Basic statistical features
        features.extend([
            np.mean(window, axis=-1),  # Mean
            np.std(window, axis=-1),   # Standard deviation
            np.var(window, axis=-1),   # Variance
            np.min(window, axis=-1),   # Minimum
            np.max(window, axis=-1),   # Maximum
            np.ptp(window, axis=-1),   # Peak-to-peak
        ])

        # Hjorth parameters
        features.extend(self._hjorth_parameters(window))

        # Zero crossings
        features.append(self._zero_crossings(window))

        # Shannon entropy
        features.append(self._shannon_entropy(window))

        return np.concatenate(features)

    def extract_frequency_features(self, window):
        """Extract frequency-domain features"""
        features = []

        # Compute FFT
        freqs = np.fft.rfftfreq(window.shape[-1], 1/self.fs)
        fft = np.fft.rfft(window, axis=-1)
        power = np.abs(fft) ** 2

        # Band powers (δ, θ, α, β, γ)
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }

        band_powers = []
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(power[..., mask], axis=-1)
            band_powers.append(band_power)

        features.extend(band_powers)

        # Band ratios
        delta, theta, alpha, beta, gamma = band_powers
        features.extend([
            alpha / (theta + 1e-8),  # α/θ ratio
            beta / (alpha + 1e-8),   # β/α ratio
            (alpha + beta) / (delta + theta + 1e-8)  # Activation index
        ])

        # Spectral centroid
        features.append(self._spectral_centroid(freqs, power))

        # Spectral roll-off
        features.append(self._spectral_rolloff(freqs, power))

        return np.concatenate(features)

    def extract_wavelet_features(self, window):
        """Extract wavelet-based features"""
        features = []

        # Use db4 wavelet
        wavelet = 'db4'
        level = 4

        for ch in range(window.shape[0]):
            channel_data = window[ch]

            # Decompose signal
            coeffs = pywt.wavedec(channel_data, wavelet, level=level)

            # Energy per level
            for coeff in coeffs:
                energy = np.sum(coeff ** 2)
                features.append(energy)

            # Coherence between channels (if multiple channels)
            if window.shape[0] > 1:
                for ch2 in range(ch + 1, window.shape[0]):
                    coherence = self._wavelet_coherence(coeffs, pywt.wavedec(window[ch2], wavelet, level=level))
                    features.extend(coherence)

        return np.array(features)

    def _hjorth_parameters(self, window):
        """Compute Hjorth parameters: activity, mobility, complexity"""
        activity = np.var(window, axis=-1)

        # First derivative
        diff1 = np.diff(window, axis=-1)
        mobility = np.sqrt(np.var(diff1, axis=-1) / (activity + 1e-8))

        # Second derivative
        diff2 = np.diff(diff1, axis=-1)
        complexity = np.sqrt(np.var(diff2, axis=-1) / (np.var(diff1, axis=-1) + 1e-8)) / mobility

        return [activity, mobility, complexity]

    def _zero_crossings(self, window):
        """Count zero crossings"""
        signs = np.sign(window)
        crossings = np.sum(np.abs(np.diff(signs, axis=-1)), axis=-1) / 2
        return crossings

    def _shannon_entropy(self, window):
        """Compute Shannon entropy"""
        # Histogram-based entropy
        hist, _ = np.histogram(window.flatten(), bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return np.full(window.shape[0], entropy)  # Same for all channels

    def _spectral_centroid(self, freqs, power):
        """Compute spectral centroid"""
        numerator = np.sum(freqs * power, axis=-1)
        denominator = np.sum(power, axis=-1)
        centroid = numerator / (denominator + 1e-8)
        return centroid

    def _spectral_rolloff(self, freqs, power, rolloff_percent=0.85):
        """Compute spectral roll-off frequency"""
        total_power = np.sum(power, axis=-1)
        cumulative_power = np.cumsum(power, axis=-1)
        rolloff_idx = np.argmax(cumulative_power >= rolloff_percent * total_power[..., None], axis=-1)
        rolloff_freq = freqs[rolloff_idx]
        return rolloff_freq

    def _wavelet_coherence(self, coeffs1, coeffs2):
        """Compute wavelet coherence between two signals"""
        coherence = []
        for c1, c2 in zip(coeffs1, coeffs2):
            # Cross-spectrum
            cross = np.abs(np.sum(c1 * np.conj(c2)))
            # Auto-spectra
            auto1 = np.sum(np.abs(c1) ** 2)
            auto2 = np.sum(np.abs(c2) ** 2)
            # Coherence
            coh = cross ** 2 / (auto1 * auto2 + 1e-8)
            coherence.append(coh.real)
        return coherence

    def __call__(self, window):
        """Extract all features from a window"""
        time_features = self.extract_time_features(window)
        freq_features = self.extract_frequency_features(window)
        wavelet_features = self.extract_wavelet_features(window)

        # Combine all features
        all_features = np.concatenate([
            time_features,
            freq_features,
            wavelet_features
        ])

        return all_features
