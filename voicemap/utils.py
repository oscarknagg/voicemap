import torch


def whiten(batch: torch.Tensor, rms: int = 0.038021) -> torch.Tensor:
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


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))
