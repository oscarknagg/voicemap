import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import argparse
from olympic.callbacks import CSVLogger, Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic.metrics import accuracy as _accuracy
from olympic import fit

from voicemap.datasets import LibriSpeech, SpeakersInTheWild, ClassConcatDataset, SpectrogramDataset
from voicemap.models import ResidualEmbedding, BaselineClassifier
from voicemap.utils import whiten, setup_dirs
from voicemap.losses import AdditiveMarginSoftmax
from voicemap.callbacks import Accuracy
from config import PATH


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--filters', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--n-seconds', type=int)
parser.add_argument('--downsampling', type=int)
parser.add_argument('--spectrogram', type=lambda x: x.lower()[0] == 't', default=True,
                    help='Whether or not to use raw waveform or a spectogram as inputs.')
parser.add_argument('--window-length', type=float, help='STFT window length in seconds.')
parser.add_argument('--window-hop', type=float, help='STFT window hop in seconds.')
args = parser.parse_args()

param_str = '__'.join([f'{k}={str(v)}' for k, v in vars(args).items()])

print(param_str)

test_fraction = 0.1

if args.spectrogram:
    in_channels = int(args.window_length * 16000) // 2 + 1
else:
    in_channels = 1


###################
# Create datasets #
###################
# librispeech_subsets = ['train-clean-100', 'train-clean-360', 'train-other-500']
librispeech_subsets = ['dev-clean']
librispeech = LibriSpeech(librispeech_subsets, args.n_seconds, args.downsampling, stochastic=True, pad=False)
sitw = SpeakersInTheWild('dev', 'enroll-core', args.n_seconds, args.downsampling, stochastic=True, pad=True)

if args.spectrogram:
    librispeech = SpectrogramDataset(librispeech, normalise=True, window_length=args.window_length,
                                     window_hop=args.window_hop)
    sitw = SpectrogramDataset(sitw, normalise=True, window_length=args.window_length, window_hop=args.window_hop)


# data = ClassConcatDataset([librispeech, sitw])
data = librispeech
num_classes = data.num_classes
print(f'Total no. speakers = {num_classes}')

unseen_speakers = LibriSpeech('dev-clean', args.n_seconds, args.downsampling, stochastic=True, pad=False)
if args.spectrogram:
    unseen_speakers = SpectrogramDataset(unseen_speakers, normalise=True, window_length=args.window_length,
                                         window_hop=args.window_hop)

indices = range(len(data))
speaker_ids = librispeech.df['speaker_id'].values.tolist() #+ sitw.df['speaker_id'].values.tolist()
train_indices, test_indices, _, _ = train_test_split(indices, speaker_ids, test_size=test_fraction,
                                                     stratify=speaker_ids)

train = torch.utils.data.Subset(data, train_indices)
test = torch.utils.data.Subset(data, test_indices)


################
# Define model #
################
# model = ResidualClassifier(in_channels, args.filters, [2, 2, 2, 2], num_classes)
model = BaselineClassifier(in_channels, args.filters, 256, num_classes)
model.to(device, dtype=torch.double)


############
# Training #
############
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()


if args.spectrogram:
    def prepare_batch(batch):
        # Normalise inputs
        # Move to GPU and convert targets to int
        x, y = batch
        return x.double().cuda(), y.long().cuda()
else:
    def prepare_batch(batch):
        # Normalise inputs
        # Move to GPU and convert targets to int
        x, y = batch
        return whiten(x).cuda(), y.long().cuda()


def gradient_step(model, optimiser, loss_fn, x, y, epoch):
    # Slight modification of regular gradient step to
    model.train()
    optimiser.zero_grad()
    y_pred = model(x, y)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


callbacks = [
    # Accuracy(test_loader, loss_fn),
    Evaluate(test_loader),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=False, min_delta=0.05),
    ModelCheckpoint(filepath=PATH + f'/models/classifier_{param_str}.pt',
                    monitor='val_loss', save_best_only=True, verbose=False),
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
    metrics=['accuracy'],
    update_fn=gradient_step
)
