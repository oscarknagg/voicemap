import torch


def whiten(batch, rms=0.038021):
    """This function whitens a batch of samples so each sample
    has 0 mean and the same root mean square amplitude i.e. volume.

    NB This function operates on a 3D torch.Tensor of shape (n_samples, 1, n_timesteps,)
    """
    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (n_samples, 1, n_timesteps,).')

    # Subtract mean
    sample_wise_mean = batch.mean(dim=2)
    whitened_batch = batch - sample_wise_mean.repeat([1, 1, batch.shape[2]]).transpose(dim0=1, dim1=0)

    # Divide through
    rescaling_factor = rms / (torch.sqrt(torch.mul(batch, batch).mean(dim=2)) + 1e-8)
    whitened_batch = whitened_batch*rescaling_factor.repeat([1, 1, batch.shape[2]]).transpose(dim0=1, dim1=0)

    return whitened_batch


def query_prototype_distances(query, prototypes, q, k):
    """Efficiently calculate matching scores between query samples and class prototypes
    in an n-shot, k-way, q-query-per-class classification task.

    The output should be a tensor of shape (q * k, k) in which each of the q * k rows
    contains the distances between that query sample and the k class prototypes.

    This is equivalent to the the logits of a k-way classification network.

    # Arguments
        query: Query samples. A tensor of shape (q * k, d) where d is the embedding dimension
        prototypes: Class prototypes. A tensor of shape (k, d) where d is the embedding dimension
    """
    distances = (
            query.unsqueeze(1).expand(q * k, k, -1) -
            prototypes.unsqueeze(0).expand(q * k, k, -1)
    ).pow(2).sum(dim=2)
    return distances
