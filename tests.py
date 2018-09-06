import unittest
import soundfile as sf
import numpy as np

from utils import whiten
from config import PATH


class TestLibriSpeechDataset(unittest.TestCase):
    pass


class TestWhitening(unittest.TestCase):
    def test_whitening(self):
        desired_rms = 0.038021

        test_data, sample_rate = sf.read(PATH + '/data/whitening_test_audio.flac')
        test_data = np.stack([test_data]*2)

        whitened = whiten(test_data, desired_rms)
        
        # Mean correct
        self.assert_(np.isclose(whitened.mean().item(), 0))

        # RMS correct
        self.assert_(np.isclose(np.sqrt(np.power(whitened[0,:], 2).mean()).item(), desired_rms))
