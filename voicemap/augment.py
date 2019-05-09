import numpy as np


def time_mask(spectrogram: np.ndarray, T: int):
    t = np.random.randint(T)
    t0 = np.random.randint(0, spectrogram.shape[1] - t)
    spectrogram[:, t0:t0 + t] = 0
    return spectrogram


def freq_mask(spectrogram: np.ndarray, F: int):
    f = np.random.randint(F)
    f0 = np.random.randint(0, spectrogram.shape[1] - f)
    spectrogram[f0:f0 + f, :] = 0
    return spectrogram


class SpecAugment:
    """Data augmentation for spectrograms.

    References:
        https://arxiv.org/pdf/1904.08779.pdf

    Args:
        n_f: Number of frequency masks.
        F: Maximum size (in bins) of frequency mask.
        n_t: Number of time masks.
        T: Maximum size of time masks.
    """
    def __init__(self, n_f: int, F: int, n_t: int, T: int):
        self.n_f = n_f
        self.F = F
        self.n_t = n_t
        self.T = T

    def __call__(self, x):
        for i in range(self.n_f):
            x = freq_mask(x, self.F)

        for i in range(self.n_t):
            x = time_mask(x, self.T)

        return x
