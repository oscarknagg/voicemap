"""
Reproduce Matching Network results of Vinyals et al
"""
import torch
import argparse

from voicemap.datasets import OmniglotDataset, MiniImageNet
from voicemap.models import get_few_shot_encoder
from voicemap.few_shot import NShotWrapper, prepare_nshot_task
from voicemap.train import fit
from voicemap.utils import query_suppport_distances
from voicemap.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--fce')
args = parser.parse_args()


if args.dataset == 'omniglot':
    n_shot_train = 1
    k_way_train = 60
    q_queries_train = 5
    n_epochs = 40
    episodes_per_epoch = 100
    n_shot_val = 1
    k_way_val = 5
    q_queries_val = 1
    evaluation_episodes = 1000
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
else:
    raise NotImplementedError


def matching_net_eposide(model, optimiser, loss_fn, x, y, **kwargs):
    if kwargs['train']:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model.enocder(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:kwargs['n_shot'] * kwargs['k_way']]
    queries = embeddings[kwargs['n_shot'] * kwargs['k_way']:]

    # Optionally apply full context embeddings
    if kwargs['fce']:
        support = model.lstm(support)

    # Efficiently calculate cosine distance between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = query_suppport_distances(queries, support, kwargs['q_queries'], kwargs['k_way'])
    logits = -distances

    # First instance is always correct one by construction so the label reflects this
    # Label is repeated by the number of queries
    loss = loss_fn(logits, y)

    # Prediction probabilities are softmax over distances
    y_pred = logits.softmax(dim=1)

    if kwargs['train']:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss.item(), y_pred