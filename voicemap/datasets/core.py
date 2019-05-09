from torch.utils.data import Dataset, Sampler
from torchvision import datasets
import torch
from abc import abstractmethod
from typing import List, Iterable
import numpy as np
import pandas as pd
import bisect


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


class DatasetFolder(datasets.DatasetFolder):
    @property
    def num_classes(self):
        return len(set(self.classes))


class ClassConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets with distinct classes.


    Args:
        datasets: List of datasets to be concatenated. Each dataset class must implement the num_classes method
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, datasets):
        super(ClassConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(len(dataset) for dataset in self.datasets)
        self.cumulative_classes = self.cumsum(dataset.num_classes for dataset in self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample, class_idx = self.datasets[dataset_idx][sample_idx]

        if dataset_idx == 0:
            class_idx = class_idx
        else:
            class_idx += self.cumulative_classes[dataset_idx - 1]

        return sample, class_idx

    @property
    def num_classes(self):
        return self.cumulative_classes[-1]


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        Args:
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


class PairDataset(Dataset):
    def __init__(self, dataset, pairs=None, labels=None, num_pairs=None):
        self.dataset = dataset
        self.pairs = pairs
        self.labels = labels
        if num_pairs is None:
            assert len(pairs) == len(labels)
        else:
            self.num_pairs = num_pairs

    def __len__(self):
        if self.num_pairs is None:
            return len(self.labels)
        else:
            return self.num_pairs

    def __getitem__(self, index):
        if self.pairs is not None:
            x = self.dataset[self.pairs[index][0]][0]
            y = self.dataset[self.pairs[index][0]][0]
            label = self.labels[index]
        else:
            index_1 = np.random.randint(len(self.dataset))
            index_2 = np.random.randint(len(self.dataset))
            x, x_label = self.dataset[index_1]
            y, y_label = self.dataset[index_2]
            label = x_label == y_label

        return (x, y), label


def collate_pairs(pairs):
    lefts = []
    rights = []
    labels = []
    for (x, y), label in pairs:
        lefts.append(x)
        rights.append(y)
        labels.append(label)

    x = torch.from_numpy(np.stack(lefts)).double()
    y = torch.from_numpy(np.stack(rights)).double()
    return (x, y), labels


class AudioDataset(Dataset):
    base_sampling_rate: int
    df: pd.DataFrame

    @property
    @abstractmethod
    def num_classes(self):
        raise NotImplementedError