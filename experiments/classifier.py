import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import argparse
from olympic.callbacks import CSVLogger, Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic import fit

from voicemap.datasets import LibriSpeech
from voicemap.models import ResidualClassifier
from voicemap.utils import whiten, setup_dirs
from config import PATH


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--filters', type=int)
parser.add_argument('--embedding-dim', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--n-seconds', type=int)
parser.add_argument('--downsampling', type=int)
args = parser.parse_args()

param_str = f'filters={args.filters}__embedding_dim={args.embedding_dim}__n_seconds={args.n_seconds}__' \
    f'downsampling={args.downsampling}'
print(param_str)

test_fraction = 0.1
# n-shot task params
num_tasks = 500
n_shot = 1
k_way = 5


###################
# Create datasets #
###################
# dataset = ['train-clean-100', 'train-clean-360', 'train-other-500']
dataset = ['dev-clean', ]
data = LibriSpeech(dataset, args.n_seconds, args.downsampling, stochastic=True, pad=False)

unseen_speakers = LibriSpeech('dev-clean', args.n_seconds, args.downsampling, stochastic=True, pad=False)

indices = range(len(data))
speaker_ids = data.df['speaker_id'].values
train_indices, test_indices, _, _ = train_test_split(indices, speaker_ids, test_size=test_fraction,
                                                     stratify=speaker_ids)

gb = data.df.iloc[train_indices].groupby('speaker_id').agg({'seconds': 'sum'})
print('TRAIN: {} unique speakers with {:.1f}+-{:.1f} seconds of audio each,'.format(
    len(gb), gb['seconds'].mean(), gb['seconds'].std()))

gb = data.df.iloc[test_indices].groupby('speaker_id').agg({'seconds': 'sum'})
print('TEST: {} unique speakers with {:.1f}+-{:.1f} seconds of audio each,'.format(
    len(gb), gb['seconds'].mean(), gb['seconds'].std()))

train = torch.utils.data.Subset(data, train_indices)
test = torch.utils.data.Subset(data, test_indices)


################
# Define model #
################
model = ResidualClassifier(args.filters, [2, 2, 2], data.num_classes())
model.to(device, dtype=torch.double)


############
# Training #
############
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def prepare_batch(batch):
    # Normalise inputs
    # Move to GPU and convert targets to int
    x, y = batch
    return whiten(x).cuda(), y.long().cuda()


def prepare_n_shot_batch(query, support):
    query = torch.from_numpy(query[0]).to(device, dtype=torch.double)
    support = torch.from_numpy(support[0]).to(device, dtype=torch.double)
    return query, support


callbacks = [
    Evaluate(test_loader),
    # Evaluate n-shot tasks on seen classes
    # NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=data_stochastic,
    #                     prepare_batch=prepare_n_shot_batch, prefix=''),
    # # Evaluate n-shot on tasks on unseen classes
    # NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=unseen_speakers,
    #                     prepare_batch=prepare_n_shot_batch, prefix='test_'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=True, min_delta=0.005),
    ModelCheckpoint(filepath=PATH + f'/models/classifier_{param_str}.pt',
                    monitor='val_categorical_accuracy'),
    CSVLogger(PATH + f'/logs/classifier_{param_str}.csv'),
]


fit(
    model,
    opt,
    loss_fn,
    epochs=50,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['accuracy']
)
