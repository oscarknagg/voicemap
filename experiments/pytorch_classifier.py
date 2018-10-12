import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from voicemap.librispeech import LibriSpeechDataset
from voicemap.models import get_classifier
from voicemap.callbacks import CSVLogger, ValidationMetrics, ReduceLROnPlateau
from voicemap.train import fit
from voicemap.utils import whiten
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')

np.random.seed(0)


##############
# Parameters #
##############
filters = 128
embedding = 64
batchsize = 64
n_seconds = 3
downsampling = 4


###################
# Create datasets #
###################
dataset = ['train-clean-100', 'train-clean-360']
data = LibriSpeechDataset(dataset, n_seconds, downsampling, stochastic=False)

indices = range(len(data))
speaker_ids = data.df['speaker_id'].values
train_indices, test_indices, _, _ = train_test_split(indices, speaker_ids, test_size=0.1, stratify=speaker_ids)

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
model = get_classifier(filters, embedding, data.num_classes())
model.to(device, dtype=torch.double)


############
# Training #
############
train_loader = DataLoader(train, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def prepare_batch(batch):
    # Normalise inputs
    # Move to GPU and convert targets to int
    x, y = batch
    return whiten(x).cuda(), y.long().cuda()


callbacks = [
    ValidationMetrics(test_loader),
    ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=5, verbose=True),
    CSVLogger(PATH + '/logs/pytorch_baseline_classifier.csv'),
]


torch.backends.cudnn.benchmark = True
fit(
    model,
    opt,
    loss_fn,
    epochs=40,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['categorical_accuracy']
)
