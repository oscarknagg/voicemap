from torch.utils.data import Dataset, Sampler
from torchvision import datasets
from abc import abstractmethod
import torch
import bisect
from typing import List, Union, Iterable, Callable
from tqdm import tqdm
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import os

from config import PATH, DATA_PATH


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


def to_categorical(y, num_classes):
    """Transforms an integer class label into a one-hot label (single integer to 1D vector)."""
    if y >= num_classes:
        raise(ValueError, 'Integer label is greater than the number of classes.')
    one_hot = np.zeros(num_classes)
    one_hot[y] = 1
    return one_hot


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
                 dataset: torch.utils.data.Dataset,
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


class LibriSpeech(AudioDataset):
    """Dataset object representing the LibriSpeech dataset (http://www.openslr.org/12/).

    # Arguments
        subsets: What LibriSpeech datasets to include.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        down_sampling:
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
        cache: bool. Whether or not to use the cached index file
    """

    base_sampling_rate = 16000

    def __init__(self,
                 subsets: Union[str, List[str]],
                 seconds: Union[int, None],
                 down_sampling: int,
                 label: str = 'speaker',
                 stochastic: bool = True,
                 pad: bool = False,
                 cache: bool = True,
                 data_path: str = DATA_PATH):
        if label not in ('sex', 'speaker'):
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')')

        if seconds is not None:
            if int(seconds * self.base_sampling_rate) % down_sampling != 0:
                raise(ValueError, 'Down sampling must be an integer divisor of the fragment length.')

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset
        if isinstance(subsets, str):
            subsets = [subsets]

        self.subsets = subsets
        self.seconds = seconds
        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.down_sampling = down_sampling
        self.stochastic = stochastic
        self.pad = pad
        self.label = label
        self.data_path = data_path

        # print('Initialising LibriSpeechDataset with minimum length = {}s and subsets = {}'.format(seconds, subsets))

        cached_df = []
        found_cache = {s: False for s in subsets}
        if cache:
            # Check for cached files
            for s in subsets:
                subset_index_path = PATH + '/data/{}.index.csv'.format(s)
                if os.path.exists(subset_index_path):
                    cached_df.append(pd.read_csv(subset_index_path))
                    found_cache[s] = True

        # Index the remaining subsets if any
        if all(found_cache.values()) and cache:
            self.df = pd.concat(cached_df)
        else:
            df = pd.read_csv(PATH +'/data/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
            df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
            df = df.assign(
                sex=df['sex'].apply(lambda x: x.strip()),
                subset=df['subset'].apply(lambda x: x.strip()),
                name=df['name'].apply(lambda x: x.strip()),
            )

            audio_files = []
            for subset, found in found_cache.items():
                if not found:
                    audio_files += self.index_subset(subset)

            # Merge individual audio files with indexing dataframe
            df = pd.merge(df, pd.DataFrame(audio_files))

            # # Concatenate with already existing dataframe if any exist
            self.df = pd.concat(cached_df+[df])

        # Save index files to data folder
        for s in subsets:
            self.df[self.df['subset'] == s].to_csv(PATH + '/data/{}.index.csv'.format(s), index=False)

        # Trim too-small files
        if not self.pad and self.seconds is not None:
            self.df = self.df[self.df['seconds'] > self.seconds]
        self.num_speakers = len(self.df['id'].unique())

        # Renaming for clarity
        self.df = self.df.rename(columns={'id': 'speaker_id', 'minutes': 'speaker_minutes'})

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']
        self.datasetid_to_sex = self.df.to_dict()['sex']

        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes)}

        print('Finished indexing data. {} usable files found.'.format(len(self)))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance)-self.fragment_length, 1))
        else:
            fragment_start_index = 0

        if self.seconds is not None:
            instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]
        else:
            # Use whole sample
            pass

        # Check for required length and pad if necessary
        if hasattr(self, 'fragment_length'):
            if self.pad and len(instance) < self.fragment_length:
                less_timesteps = self.fragment_length - len(instance)
                if self.stochastic:
                    # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                    # before and the appropriate number of 0s after the instance
                    before_len = np.random.randint(0, less_timesteps)
                    after_len = less_timesteps - before_len

                    instance = np.pad(instance, (before_len, after_len), 'constant')
                else:
                    # Deterministic padding. Append 0s to reach self.fragment_length
                    instance = np.pad(instance, (0, less_timesteps), 'constant')

        if self.label == 'sex':
            sex = self.datasetid_to_sex[index]
            label = sex_to_label[sex]
        elif self.label == 'speaker':
            label = self.datasetid_to_speaker_id[index]
            label = self.speaker_id_mapping[label]
        else:
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')'.format(self.label))

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.down_sampling]

        return instance, label

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    def index_subset(self, subset):
        """Index a subset by looping through all of it's files and recording their speaker ID, filepath and length.

        # Arguments
            subset: Name of the subset

        # Returns
            audio_files: A list of dicts containing information about all the audio files in a particular subset of the
            LibriSpeech dataset
        """
        audio_files = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(self.data_path + '/LibriSpeech/{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.flac')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(self.data_path + '/LibriSpeech/{}/'.format(subset)):
            if len(files) == 0:
                continue

            librispeech_id = int(root.split('/')[-2])

            for f in files:
                # Skip non-sound files
                if not f.endswith('.flac'):
                    continue

                progress_bar.update(1)

                instance, samplerate = sf.read(os.path.join(root, f))

                audio_files.append({
                    'id': librispeech_id,
                    'filepath': os.path.join(root, f),
                    'length': len(instance),
                    'seconds': len(instance) * 1. / self.base_sampling_rate
                })

        progress_bar.close()
        return audio_files


class SpeakersInTheWild(AudioDataset):
    """Dataset class representing the Speakers in the wild dataset (http://www.speech.sri.com/projects/sitw/).

    # Arguments
        subsets: What subset of the SitW dataset to use.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        down_sampling:
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
    """

    base_sampling_rate = 16000

    def __init__(self,
                 split: str,
                 subset: Union[str, List[str]],
                 seconds: Union[int, None],
                 down_sampling: int,
                 stochastic: bool = True,
                 pad: bool = False,
                 data_path: str = DATA_PATH):
        self.split = split
        self.seconds = seconds
        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.down_sampling = down_sampling
        self.stochastic = stochastic
        self.pad = pad
        self.data_path = data_path

        # Get dataset info
        self.df = pd.read_csv(self.data_path + f'/sitw/{self.split}/lists/{subset}.lst',
                              delimiter=' ', names=['id', 'filepath'])

        # Have to use /keys/meta.list to get speaker_id
        meta_names = ['filepath', 'speaker_id', 'gender', 'mic_type', 'session_id', 'audio_start', 'audio_end',
                      'num_speakers', 'artifact_labels', 'artifact_level', 'environment', 'tag1', 'tag2', 'tag3',
                      'tag4']

        # Eval set has a greater number of tags
        if self.split == 'eval':
            meta_names += ['tag5', 'tag6', 'tag7']

        meta = pd.read_csv(self.data_path + f'/sitw/{self.split}/keys/meta.lst', delimiter=' ', names=meta_names)
        self.df = self.df.merge(meta, on='filepath')
        self.df['filepath'] = self.data_path + f'/sitw/{self.split}/' + self.df['filepath']
        self.df['index'] = self.df.index.values

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']

        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes)}

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance) - self.fragment_length, 1))
        else:
            fragment_start_index = 0

        if self.seconds is not None:
            instance = instance[fragment_start_index:fragment_start_index + self.fragment_length]
        else:
            # Use whole sample
            pass

        if hasattr(self, 'fragment_length'):
            # Check for required length and pad if necessary
            if self.pad and len(instance) < self.fragment_length:
                less_timesteps = self.fragment_length - len(instance)
                if self.stochastic:
                    # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                    # before and the appropriate number of 0s after the instance
                    less_timesteps = self.fragment_length - len(instance)

                    before_len = np.random.randint(0, less_timesteps)
                    after_len = less_timesteps - before_len

                    instance = np.pad(instance, (before_len, after_len), 'constant')
                else:
                    # Deterministic padding. Append 0s to reach self.fragment_length
                    instance = np.pad(instance, (0, less_timesteps), 'constant')

        label = self.datasetid_to_speaker_id[index]
        label = self.speaker_id_mapping[label]

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.down_sampling]

        return instance, label


class CommonVoice(AudioDataset):
    base_sampling_rate = 48000

    def __init__(self,
                 language: str,
                 seconds: Union[int, None],
                 sampling_rate: int = None,
                 stochastic: bool = True,
                 pad: bool = True,
                 data_path: str = DATA_PATH):
        self.language = language
        self.seconds = seconds
        self.data_path = data_path
        if sampling_rate is None:
            self.sampling_rate = self.base_sampling_rate
        else:
            if sampling_rate > self.base_sampling_rate:
                raise ValueError('Shouldn\'t have sampling rate higher than the sampling rate of the raw data.')
            self.sampling_rate = sampling_rate

        if seconds is not None:
            self.fragment_length = int(seconds * self.base_sampling_rate)
        self.stochastic = stochastic
        self.pad = pad

        self.df = pd.read_csv(self.data_path + f'/CommonVoice/{self.language}/validated.csv')
        self.df['speaker_id'] = self.df['client_id']
        self.df['filepath'] = self.data_path + f'/CommonVoice/{self.language}/clips/' + self.df['path'] + '.mp3'

        self.df['index'] = self.df.index.values

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']

        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes)}

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    def __getitem__(self, index):
        instance, samplerate = librosa.core.load(self.datasetid_to_filepath[index], sr=self.sampling_rate)
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance) - self.fragment_length, 1))
        else:
            fragment_start_index = 0

        if self.seconds is not None:
            instance = instance[fragment_start_index:fragment_start_index + self.fragment_length]
        else:
            # Use whole sample
            pass

        if hasattr(self, 'fragment_length'):
            # Check for required length and pad if necessary
            if self.pad and len(instance) < self.fragment_length:
                less_timesteps = self.fragment_length - len(instance)
                if self.stochastic:
                    # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                    # before and the appropriate number of 0s after the instance
                    less_timesteps = self.fragment_length - len(instance)

                    before_len = np.random.randint(0, less_timesteps)
                    after_len = less_timesteps - before_len

                    instance = np.pad(instance, (before_len, after_len), 'constant')
                else:
                    # Deterministic padding. Append 0s to reach self.fragment_length
                    instance = np.pad(instance, (0, less_timesteps), 'constant')

        label = self.datasetid_to_speaker_id[index]
        label = self.speaker_id_mapping[label]

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.down_sampling]

        return instance, label


class DummyDataset(Dataset):
    def __init__(self, samples_per_class: int, num_classes: int, n_features: int = 2):
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.num_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.num_classes

    def __getitem__(self, item):
        class_id = item % self.num_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)
