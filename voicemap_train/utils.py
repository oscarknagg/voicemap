import torch
import shutil

from config import PATH


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


def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/models/')
