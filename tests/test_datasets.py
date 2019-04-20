import unittest
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler

from voicemap_train.datasets import *
from voicemap_train.few_shot import NShotWrapper, NShotSampler, create_nshot_task_label


class TestLibriSpeechDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = LibriSpeechDataset('dev-clean', 3, 4)

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

        for i in range(0, n * k, n):
            support_set_classes_correct = np.all(support_set_labels[i:i + n] == support_set_labels[i])
            self.assertTrue(
                support_set_classes_correct,
                'Classes of support set samples should be arranged like: [class_1]*n + [class_2]*n + ... + [class_k]*n'
            )


class TestOmniglotDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = OmniglotDataset('evaluation')

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

        for i in range(0, n * k, n):
            support_set_classes_correct = np.all(support_set_labels[i:i + n] == support_set_labels[i])
            self.assertTrue(
                support_set_classes_correct,
                'Classes of support set samples should be arranged like: [class_1]*n + [class_2]*n + ... + [class_k]*n'
            )


class TestNShot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset(samples_per_class=1000, n_classes=20)

    def test_n_shot_sampler(self):
        n, k, q = 2, 4, 3
        n_shot_taskloader = DataLoader(self.dataset,
                                       batch_sampler=NShotSampler(self.dataset, 100, n, k, q))

        # Load a single n-shot task and check it's properties
        for x, y in n_shot_taskloader:
            support = x[:n*k]
            queries = x[n*k:]
            support_labels = y[:n*k]
            query_labels = y[n*k:]

            # Check ordering of support labels is correct
            for i in range(0, n * k, n):
                support_set_labels_correct = torch.all(support_labels[i:i + n] == support_labels[i])
                self.assertTrue(
                    support_set_labels_correct,
                    'Classes of support set samples should be arranged like: '
                    '[class_1]*n + [class_2]*n + ... + [class_k]*n'
                )

            # Check ordering of query labels is correct
            for i in range(0, q * k, q):
                support_set_labels_correct = torch.all(query_labels[i:i + q] == query_labels[i])
                self.assertTrue(
                    support_set_labels_correct,
                    'Classes of query set samples should be arranged like: '
                    '[class_1]*q + [class_2]*q + ... + [class_k]*q'
                )

            # Check labels are consistent across query and support
            for i in range(k):
                self.assertEqual(
                    support_labels[i*n],
                    query_labels[i*q],
                    'Classes of query and support set should be consistent.'
                )

            # Check no overlap of IDs between support and query.
            # By construction the first feature in the DummyDataset is the
            # id of the sample in the dataset so we can use this to test
            # for overlap betwen query and suppport samples
            self.assertEqual(
                len(set(support[:, 0].numpy()).intersection(set(queries[:, 0].numpy()))),
                0,
                'There should be no overlap between support and query set samples.'
            )

            break
