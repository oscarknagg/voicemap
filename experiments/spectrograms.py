"""Preprocesses waveforms to create spectogram datasets."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, SpectrogramDataset
from voicemap.utils import mkdir
from config import DATA_PATH


window_length = 0.02
window_hop = 0.01
norm = 'global'


def generate_filepath(dataset, subset, speaker_id, index):
    out = DATA_PATH + f'/{dataset}.spec/{subset}/{speaker_id}/{index}.npy'
    return out


# librispeech_subsets = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360', 'train-other-500']
librispeech_subsets = ['train-other-500']

dataset = 'LibriSpeech'
for subset in librispeech_subsets:

    print(subset)
    waveforms = LibriSpeech(subset, seconds=None, down_sampling=1, stochastic=False, pad=False)
    spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
    loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

    mkdir(DATA_PATH + f'/{dataset}.spec/')
    mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/')

    pbar = tqdm(total=len(loader))
    for i, (spec, y) in enumerate(loader):
        spec = spec.numpy()
        y = y.item()
        outpath = generate_filepath(dataset, subset, y, i)
        mkdir(DATA_PATH + f'/{dataset}.spec/{subset}/{y}/')

        np.save(outpath, spec)
        pbar.update(1)

    pbar.close()

dataset = 'sitw'
for split in ['dev', 'test']:
    print(split)
    waveforms = SpeakersInTheWild(split, 'enroll-core', seconds=None, down_sampling=1, stochastic=False, pad=False)
    spectrograms = SpectrogramDataset(waveforms, normalisation=norm, window_length=window_length, window_hop=window_hop)
    loader = DataLoader(spectrograms, batch_size=1, shuffle=False)

    mkdir(DATA_PATH + f'/{dataset}.spec/')
    mkdir(DATA_PATH + f'/{dataset}.spec/{split}/')

    pbar = tqdm(total=len(loader))
    for i, (spec, y) in enumerate(loader):
        spec = spec.numpy()
        y = y.item()
        outpath = generate_filepath(dataset, split, y, i)
        mkdir(DATA_PATH + f'/{dataset}.spec/{split}/{y}/')

        np.save(outpath, spec)
        pbar.update(1)

    pbar.close()
