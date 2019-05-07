import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import argparse
from olympic.callbacks import CSVLogger, Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic import fit

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, ClassConcatDataset
from voicemap.models import ResidualClassifier, get_classifier
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
parser.add_argument('--lr', type=float)
parser.add_argument('--l2', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--filters', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--n-seconds', type=int)
parser.add_argument('--downsampling', type=int)
args = parser.parse_args()

param_str = '__'.join([f'{k}={str(v)}' for k, v in vars(args).items()])

print(param_str)

test_fraction = 0.1


###################
# Create datasets #
###################
librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500']
librispeech = LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, stochastic=True, pad=False)
sitw = SpeakersInTheWild('dev', 'enroll-core', args.n_seconds, args.downsampling, True, True)

data = ClassConcatDataset([librispeech, sitw])
num_classes = data.num_classes
print(f'Total no. speakers = {num_classes}')

unseen_speakers = LibriSpeech('dev-clean', args.n_seconds, args.downsampling, stochastic=True, pad=False)

indices = range(len(data))
speaker_ids = librispeech.df['speaker_id'].values.tolist() + sitw.df['speaker_id'].values.tolist()
train_indices, test_indices, _, _ = train_test_split(indices, speaker_ids, test_size=test_fraction,
                                                     stratify=speaker_ids)

train = torch.utils.data.Subset(data, train_indices)
test = torch.utils.data.Subset(data, test_indices)


################
# Define model #
################
model = ResidualClassifier(args.filters, [2, 2, 2, 2], num_classes)
model.to(device, dtype=torch.double)


############
# Training #
############
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
loss_fn = nn.CrossEntropyLoss().cuda()


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
    ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=True, min_delta=0.005),
    ModelCheckpoint(filepath=PATH + f'/models/classifier_{param_str}.pt',
                    monitor='val_accuracy', save_best_only=True, verbose=True),
    CSVLogger(PATH + f'/logs/classifier_{param_str}.csv'),
]

fit(
    model,
    opt,
    loss_fn,
    epochs=args.epochs,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['accuracy']
)
