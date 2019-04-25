import unittest
import soundfile as sf
import numpy as np
import torch

from voicemap.utils import *
from config import PATH, DATA_PATH


class TestWhitening(unittest.TestCase):
    def test_whitening_no_batch(self):
        desired_rms = 0.038021

        test_data, sample_rate = sf.read(DATA_PATH + '/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
        test_data = np.stack([test_data]*2)
        test_data = test_data[:, np.newaxis, :]

        whitened = whiten(torch.from_numpy(test_data), desired_rms)
        # Mean correct
        self.assertTrue(
            np.isclose(whitened.mean().item(), 0),
            'Whitening should reduce mean to 0.'
        )

        # RMS correct
        self.assertTrue(
            np.isclose(np.sqrt(np.power(whitened[0, :], 2).mean()).item(), desired_rms),
            'Whitening should change RMS to desired value.'
        )
