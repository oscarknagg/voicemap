from config import PATH, LIBRISPEECH_SAMPLING_RATE
from keras.utils import Sequence
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import json
import os


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


class LibriSpeechDataset(Sequence):
    def __init__(self, subsets, seconds, label='speaker', stochastic=True, cache=True):
        """
        This class subclasses the torch Dataset object. The __getitem__ function will return a raw audio sample and it's
        label.
        :param subsets: What LibriSpeech datasets to use
        :param seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored
        :param label: Whether to use sex or speaker ID as a label
        :param stochastic: If True then we will take a random fragment from each file of sufficient length. If False we
        wil always take a fragment starting at the beginning of a file.
        """
        assert isinstance(seconds, (int, long)), 'Length is not an integer!'
        assert label in ('sex', 'speaker'), 'Label type must be one of (\'sex\', \'speaker\')'
        self.subset = subsets
        self.fragment_seconds = seconds
        self.fragment_length = int(seconds * LIBRISPEECH_SAMPLING_RATE)
        self.stochastic = stochastic
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
        if all(found_cache.keys()) and cache:
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
            for subset, found in found_cache.iteritems():
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
        self.df = self.df[self.df['seconds'] >= self.fragment_seconds]
        self.n_files = len(self.df)

        # Index of dataframe has direct correspondence to item in dataset
        self.df.reset_index(drop=True)

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['id']
        self.datasetid_to_sex = self.df.to_dict()['filepath']

        print('Finished indexing data. {} usable files found.'.format(self.n_files))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, len(instance)-self.fragment_length)
        else:
            fragment_start_index = 0
        instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]
        if self.label == 'sex':
            sex = self.datasetid_to_sex[index]
            label = sex_to_label[sex]
        elif self.label == 'speaker':
            label = self.datasetid_to_speaker_id[index]
        else:
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')'.format(self.label))

        return instance, label

    def __len__(self):
        return self.n_files

    def build_verification_batch(self, batchsize):
        """
        This function builds a batch of verification task samples meant to be input into a siamese network. Each sample
        is two instances of the dataset retrieved with the __getitem__ function and a label which indicates whether the
        instances belong to the same speaker or not. Each batch is 50% pairs of instances from the same speaker and 50%
        pairs of instances from different speakers.
        :param batchsize: Number of verification task samples to build the batch out of.
        :return: Inputs for both sides of the siamese network and outputs indicating whether they are from the same
        speaker or not.
        """
        alike_pairs = pd.merge(
            self.df.sample(batchsize, weights='length'),
            self.df,
            on='id'
        ).sample(batchsize / 2)[['id', 'dataset_id_x', 'dataset_id_y']]

        input_1_alike = np.stack([self[i][0] for i in alike_pairs['dataset_id_x'].values])
        input_2_alike = np.stack([self[i][0] for i in alike_pairs['dataset_id_y'].values])

        x = self.df.sample(batchsize / 2, weights='length')
        y = self.df[~self.df['id'].isin(x['id'])].sample(batchsize / 2, weights='length')

        input_1_different = np.stack([self[i][0] for i in x['dataset_id'].values])
        input_2_different = np.stack([self[i][0] for i in y['dataset_id'].values])

        input_1 = np.vstack([input_1_alike, input_1_different])[:, :, np.newaxis]
        input_2 = np.vstack([input_2_alike, input_2_different])[:, :, np.newaxis]

        outputs = np.append(np.zeros(batchsize/2), np.ones(batchsize/2))[:, np.newaxis]

        return [input_1, input_2], outputs

    def build_oneshot_task(self):
        pass

    def index_subset(self, subset):
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
                    # 'dataset_id': self.datasetid,
                    'filepath': os.path.join(root, f),
                    'length': len(instance),
                    'seconds': len(instance) * 1. / LIBRISPEECH_SAMPLING_RATE
                })

        progress_bar.close()
        return audio_files


# test = LibriSpeechDataset(['dev-clean'], 3, cache=False, label='speaker')
#
# print test[0]