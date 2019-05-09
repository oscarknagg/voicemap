from typing import Union, List

import numpy as np
import pandas as pd
import soundfile as sf

from config import DATA_PATH


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