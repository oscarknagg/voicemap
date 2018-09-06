import numpy as np


def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
    whitened_batch = batch - np.tile(sample_wise_mean, (batch.shape[1], 1)).T

    # Divide through
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (batch.shape[1], 1)).T
    return whitened_batch
