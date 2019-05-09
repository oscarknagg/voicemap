from typing import Union, Callable

import librosa
import numpy as np
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Wraps a waveform dataset to transform it into a spectrogram dataset.

    Args:
        dataset: Base audio dataset.
        normalise: If True normalise output spectogram to mean 0 and variance 1.
        window_length: Length of STFT window in seconds.
        window_hop: Time in seconds between STFT windows.
        window_type: Type of STFT window i.e. 'hamming'
    """
    def __init__(self,
                 dataset: AudioDataset,
                 normalisation: Union[str, None],
                 window_length: float,
                 window_hop: float,
                 window_type: Union[str, float, tuple, Callable] = 'hamming'):
        self.dataset = dataset
        self.normalisation = normalisation
        self.window_length = window_length
        self.window_hop = window_hop
        self.window_type = window_type

        self.df = self.dataset.df

    def __len__(self):
        return len(self.dataset)

    @property
    def num_classes(self):
        return self.dataset.num_classes

    def waveform_to_logmelspectrogam(self, waveform: np.ndarray):
        D = librosa.stft(waveform,
                         n_fft=int(self.dataset.base_sampling_rate * self.window_length),
                         hop_length=int(self.dataset.base_sampling_rate * self.window_hop),
                         window=self.window_type)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        if self.normalisation == 'global':
            spect = (spect - spect.mean()) / (spect.std() + 1e-6)
        elif self.normalisation == 'frequency':
            spect = (spect - spect.mean(axis=0, keepdims=True)) / (spect.std(axis=0, keepdims=True) + 1e-6)

        return spect

    def __getitem__(self, item):
        waveform, label = self.dataset[item]
        spectrogram = self.waveform_to_logmelspectrogam(waveform[0])
        return spectrogram, label