from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import os

from config import PATH, LIBRISPEECH_SAMPLING_RATE


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


def to_categorical(y, num_classes):
    """Transforms an integer class label into a one-hot label (single integer to 1D vector)."""
    if y >= num_classes:
        raise(ValueError, 'Integer label is greater than the number of classes.')
    one_hot = np.zeros(num_classes)
    one_hot[y] = 1
    return one_hot


class LibriSpeechDataset(Dataset):
    """This class subclasses the torch.utils.data.Dataset object. The __getitem__ function will return a raw audio
    sample and it's label.

    This class also contains functionality to build verification tasks and n-shot, k-way classification tasks.

    # Arguments
        subsets: What LibriSpeech datasets to include.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        downsampling:
        label: One of {speaker, sex}. Whether to use sex or speaker ID as a label.
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
        cache: bool. Whether or not to use the cached index file
    """
    def __init__(self, subsets, seconds, downsampling, label='speaker', stochastic=True, pad=False, cache=True):
        if label not in ('sex', 'speaker'):
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')')

        if int(seconds * LIBRISPEECH_SAMPLING_RATE) % downsampling != 0:
            raise(ValueError, 'Down sampling must be an integer divisor of the fragment length.')

        self.subset = subsets
        self.fragment_seconds = seconds
        self.downsampling = downsampling
        self.fragment_length = int(seconds * LIBRISPEECH_SAMPLING_RATE)
        self.stochastic = stochastic
        self.pad = pad
        self.label = label

        print('Initialising LibriSpeechDataset with minimum length = {}s and subsets = {}'.format(seconds, subsets))

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset
        if isinstance(subsets, str):
            subsets = [subsets]

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
            df = pd.read_csv(PATH+'/data/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
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
        if not self.pad:
            self.df = self.df[self.df['seconds'] > self.fragment_seconds]
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
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes())}

        print('Finished indexing data. {} usable files found.'.format(len(self)))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance)-self.fragment_length, 1))
        else:
            fragment_start_index = 0

        instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]

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

        if self.label == 'sex':
            sex = self.datasetid_to_sex[index]
            label = sex_to_label[sex]
        elif self.label == 'speaker':
            label = self.datasetid_to_speaker_id[index]
            label = self.speaker_id_mapping[label]
        else:
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')'.format(self.label))

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.downsampling]

        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    def get_alike_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to the same speaker."""
        alike_pairs = pd.merge(
            self.df.sample(num_pairs*2, weights='length'),
            self.df,
            on='speaker_id'
        ).sample(num_pairs)[['speaker_id', 'id_x', 'id_y']]

        alike_pairs = zip(alike_pairs['id_x'].values, alike_pairs['id_y'].values)

        return alike_pairs

    def get_differing_pairs(self, num_pairs):
        """Generates a list of 2-tuples containing pairs of dataset IDs belonging to different speakers."""
        # First get a random sample from the dataset and then get a random sample from the remaining part of the dataset
        # that doesn't contain any speakers from the first random sample
        random_sample = self.df.sample(num_pairs, weights='length')
        random_sample_from_other_speakers = self.df[~self.df['speaker_id'].isin(
            random_sample['speaker_id'])].sample(num_pairs, weights='length')

        differing_pairs = zip(random_sample['id'].values, random_sample_from_other_speakers['id'].values)

        return differing_pairs

    def build_verification_batch(self, batchsize):
        """
        This method builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        alike_pairs = self.get_alike_pairs(batchsize / 2)

        # Take only the instances not labels and stack to form a batch of pairs of instances from the same speaker
        input_1_alike = np.stack([self[i][0] for i in zip(*alike_pairs)[0]])
        input_2_alike = np.stack([self[i][0] for i in zip(*alike_pairs)[1]])

        differing_pairs = self.get_differing_pairs(batchsize / 2)

        # Take only the instances not labels and stack to form a batch of pairs of instances from different speakers
        input_1_different = np.stack([self[i][0] for i in zip(*differing_pairs)[0]])
        input_2_different = np.stack([self[i][0] for i in zip(*differing_pairs)[1]])

        input_1 = np.vstack([input_1_alike, input_1_different])[:, :, np.newaxis]
        input_2 = np.vstack([input_2_alike, input_2_different])[:, :, np.newaxis]

        outputs = np.append(np.zeros(batchsize/2), np.ones(batchsize/2))[:, np.newaxis]

        return [input_1, input_2], outputs

    def yield_verification_batches(self, batchsize):
        """Convenience function to yield verification batches forever."""
        while True:
            ([input_1, input_2], labels) = self.build_verification_batch(batchsize)
            yield ([input_1, input_2], labels)

    def build_n_shot_task(self, k, n=1):
        """
        This method builds a k-way n-shot classification task. It returns a support set of n audio samples each from k
        unique speakers. In addition it will return a query sample. Downstream models will attempt to match the query
        sample to the correct samples in the support set.
        :param k: Number of unique speakers to include in this task
        :param n: Number of audio samples to include from each speaker
        :return:
        """
        if k >= self.num_speakers:
            raise(ValueError, 'k must be smaller than the number of unique speakers in this dataset!')

        if k <= 1:
            raise(ValueError, 'k must be greater than or equal to one!')

        query = self.df.sample(1, weights='length')
        query_sample = self[query.index.values[0]]
        # Add batch dimension
        query_sample = (query_sample[0][np.newaxis, :, :], query_sample[1])

        is_query_speaker = self.df['speaker_id'] == query['speaker_id'].values[0]
        not_same_sample = self.df.index != query.index.values[0]
        correct_samples = self.df[is_query_speaker & not_same_sample].sample(n, weights='length')

        # Sample k-1 speakers
        # TODO: weight by length here
        other_support_set_speakers = np.random.choice(
            self.df[~is_query_speaker]['speaker_id'].unique(), k-1, replace=False)

        other_support_samples = []
        for i in range(k-1):
            is_same_speaker = self.df['speaker_id'] == other_support_set_speakers[i]
            other_support_samples.append(
                self.df[~is_query_speaker & is_same_speaker].sample(n, weights='length')
            )
        support_set = pd.concat([correct_samples]+other_support_samples)
        support_set_samples = tuple(np.stack(i) for i in zip(*[self[i] for i in support_set.index]))

        return query_sample, support_set_samples

    @staticmethod
    def index_subset(subset):
        """
        Index a subset by looping through all of it's files and recording their speaker ID, filepath and length.
        :param subset: Name of the subset
        :return: A list of dicts containing information about all the audio files in a particular subset of the
        LibriSpeech dataset
        """
        audio_files = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(PATH + '/data/LibriSpeech/{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.flac')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(PATH + '/data/LibriSpeech/{}/'.format(subset)):
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
                    'seconds': len(instance) * 1. / LIBRISPEECH_SAMPLING_RATE
                })

        progress_bar.close()
        return audio_files


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def build_n_shot_task(self, k, n=1, query=1):
        """
        This method builds a k-way n-shot classification task. It returns a support set of n audio samples each from k
        unique speakers. In addition it will return a query sample. Downstream models will attempt to match the query
        sample to the correct samples in the support set.
        :param k: Number of unique speakers to include in this task
        :param n: Number of audio samples to include from each speaker
        :param query: Number of query samples
        :return:
        """
        if k >= self.num_classes():
            raise(ValueError, 'k must be smaller than the number of unique speakers in this dataset!')

        if k <= 1:
            raise(ValueError, 'k must be greater than or equal to one!')

        query = self.df.sample(query)
        query_samples = self[query['id'].values[0]]
        # Add batch dimension
        query_samples = (query_samples[0][np.newaxis, :, :], query_samples[1])

        is_query_character = self.df['class_id'] == query['class_id'].values[0]
        not_query_sample = ~self.df.index.isin(query['id'].values)
        correct_samples = self.df[is_query_character & not_query_sample].sample(n)

        # Sample k-1 speakers
        other_support_set_characters = np.random.choice(
            self.df[~is_query_character]['class_id'].unique(), k-1, replace=False)

        other_support_samples = []
        for i in range(k-1):
            is_same_speaker = self.df['class_id'] == other_support_set_characters[i]
            other_support_samples.append(
                self.df[~is_query_character & is_same_speaker].sample(n)
            )
        support_set = pd.concat([correct_samples]+other_support_samples)
        support_set_samples = tuple(np.stack(i) for i in zip(*[self[i] for i in support_set.index]))

        return query_samples, support_set_samples

    @staticmethod
    def index_subset(subset):
        """
        Index a subset by looping through all of it's files and recording their speaker ID, filepath and length.
        :param subset: Name of the subset
        :return: A list of dicts containing information about all the audio files in a particular subset of the
        LibriSpeech dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(PATH + '/data/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(PATH + '/data/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])

        # This performs center cropping, resizing and permuting dimensions to channels first
        instance = self.transform(instance)

        label = self.datasetid_to_class_id[item]

        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(PATH + '/data/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(PATH + '/data/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images
