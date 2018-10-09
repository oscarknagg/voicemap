import torch


def whiten(batch, rms=0.038021):
    """This function whitens a batch of samples so each sample
    has 0 mean and the same root mean square amplitude i.e. volume.

    NB This function operates on a 3D torch.Tensor of shape (n_samples, n_timesteps, 1)
    """
    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (n_samples, n_timesteps, 1).')

    # Subtract mean
    sample_wise_mean = batch.mean(dim=1)
    whitened_batch = batch - sample_wise_mean.repeat([batch.shape[1], 1, 1]).transpose(dim0=1, dim1=0)

    # Divide through
    rescaling_factor = rms / torch.sqrt(torch.mul(batch, batch).mean(dim=1))
    whitened_batch = whitened_batch*rescaling_factor.repeat([batch.shape[1], 1, 1]).transpose(dim0=1, dim1=0)

    return whitened_batch


def whiten_old(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    # Subtract mean
    sample_wise_mean = batch.mean(dim=1)
    whitened_batch = batch-sample_wise_mean.repeat([batch.shape[1], 1]).transpose(dim0=1, dim1=0)

    # Divide through
    rescaling_factor = rms / torch.sqrt(torch.mul(batch, batch).mean(dim=1))
    whitened_batch = whitened_batch*rescaling_factor.repeat([batch.shape[1], 1]).transpose(dim0=1, dim1=0)
    return whitened_batch