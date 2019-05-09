import unittest
import numpy as np

import voicemap.datasets.dummy
import voicemap.datasets.librispeech
from voicemap import datasets
from config import DATA_PATH


class TestClassConcatDataset(unittest.TestCase):
    def test_dataset(self):
        data1 = voicemap.datasets.dummy.DummyDataset(1, 10)
        data2 = voicemap.datasets.dummy.DummyDataset(1, 5)

        data = datasets.ClassConcatDataset([data1, data2])

        self.assertEqual(data.num_classes, data1.num_classes + data2.num_classes)

        class_indicies = []
        for i, (x, y) in enumerate(data):
            print(i, (x.shape, y))
            class_indicies.append(y)

        self.assertEqual(min(class_indicies), 0)
        self.assertEqual(max(class_indicies), data.num_classes - 1)


class TestPrecomputeSpectrograms(unittest.TestCase):
    """Need to precompute spectograms first"""
    def tes_dataset_sizes(self):
        librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'test-clean']
        for subset in librispeech_subsets:
            raw = voicemap.datasets.librispeech.LibriSpeech(subset, None, 1)
            preprocessed = datasets.DatasetFolder(DATA_PATH + f'/LibriSpeech.spec/{subset}/', extensions=['.npy'],
                                                  loader=np.load)
            self.assertEqual(len(raw), len(preprocessed))
            self.assertEqual(raw.num_classes, preprocessed.num_classes)
