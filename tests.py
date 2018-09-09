import unittest
import soundfile as sf
import numpy as np
import pandas as pd

from utils import whiten
from librispeech import LibriSpeechDataset
from config import PATH


class TestLibriSpeechDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = LibriSpeechDataset('dev-clean', 3)

    def test_verification_batch(self):
        # I do not test the verification batch directly but I test the funcions that generate the dataset indexes
        # of alike and different pairs
        alike_pairs = self.dataset.get_alike_pairs(16)
        self.assertTrue(
            all(self.dataset[i][1] == self.dataset[j][1] for i, j in alike_pairs),
            'All alike pairs must come from the same speaker.'
        )

        differing_pairs = self.dataset.get_differing_pairs(16)
        self.assertTrue(
            all(self.dataset[i][1] != self.dataset[j][1] for i, j in differing_pairs),
            'All differing pairs must come from different speakers.'
        )

    def test_n_shot_task(self):
        # Build a 5 way, 1 shot task
        n, k = 1, 5
        query_sample, support_set_samples = self.dataset.build_n_shot_task(k, n)
        query_label = query_sample[1]
        support_set_labels = support_set_samples[1]

        self.assertTrue(
            query_label == support_set_labels[0],
            'The first sample in the support set should be from the same speaker as the query sample.'
        )

        self.assertTrue(
            query_label not in support_set_labels[1:],
            'The query speaker should not appear anywhere in the support set except the first sample.'
        )

        self.assertTrue(
            len(np.unique(support_set_labels)) == k,
            'A k-way classification task should contain k unique speakers.'
        )

        # Build a 5 way, 5 shot task
        n, k = 5, 5
        query_sample, support_set_samples = self.dataset.build_n_shot_task(k, n)
        support_set_labels = support_set_samples[1]

        self.assertTrue(
            all(pd.value_counts(support_set_labels) == 5),
            'An n-shot task should contain n samples from each speaker.'
        )


class TestWhitening(unittest.TestCase):
    def test_whitening_no_batch(self):
        desired_rms = 0.038021

        test_data, sample_rate = sf.read(PATH + '/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
        test_data = np.stack([test_data]*2)
        test_data = test_data[:, :, np.newaxis]

        whitened = whiten(test_data, desired_rms)
        # Mean correct
        self.assertTrue(
            np.isclose(whitened.mean().item(), 0),
            'Whitening should reduce mean to 0.'
        )

        # RMS correct
        self.assertTrue(
            np.isclose(np.sqrt(np.power(whitened[0,:], 2).mean()).item(), desired_rms),
            'Whitening should change RMS to desired value.'
        )
