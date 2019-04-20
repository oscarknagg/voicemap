import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from voicemap_train.datasets import LibriSpeechDataset
from voicemap_train.models import get_classifier
from voicemap_train.callbacks import CSVLogger, EvaluateMetrics, ReduceLROnPlateau, ModelCheckpoint, NShotTaskEvaluation
from voicemap_train.train import fit
from voicemap_train.utils import whiten
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
stochastic_train = True
stochastic_test = False
test_fraction = 0.1
# n-shot task params
num_tasks = 500
n_shot = 1
k_way = 5


###################
# Create datasets #
###################
dataset = ['train-clean-100', 'train-clean-360']
data = LibriSpeechDataset(dataset, n_seconds, downsampling, stochastic=False)
data_stochastic = LibriSpeechDataset(dataset, n_seconds, downsampling, stochastic=True, pad=False)

unseen_speakers = LibriSpeechDataset('dev-clean', n_seconds, downsampling, stochastic=True, pad=False)

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

if stochastic_train:
    train = torch.utils.data.Subset(data_stochastic, train_indices)
else:
    train = torch.utils.data.Subset(data, train_indices)
if stochastic_test:
    test = torch.utils.data.Subset(data_stochastic, test_indices)
else:
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


def prepare_n_shot_batch(query, support):
    query = torch.from_numpy(query[0]).to(device, dtype=torch.double)
    support = torch.from_numpy(support[0]).to(device, dtype=torch.double)
    return query, support


callbacks = [
    EvaluateMetrics(test_loader),
    # Evaluate n-shot tasks on seen classes
    NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=data_stochastic,
                        prepare_batch=prepare_n_shot_batch, prefix=''),
    # Evaluate n-shot on tasks on unseen classes
    NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=unseen_speakers,
                        prepare_batch=prepare_n_shot_batch, prefix='test_'),
    ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=5, verbose=True, min_delta=0.005),
    ModelCheckpoint(filepath=PATH + '/models/baseline_classifier_stochastic=True_r=1.torch',
                    monitor='val_categorical_accuracy'),
    CSVLogger(PATH + '/logs/baseline_classifier_stochastic=True_r=1.csv'),
]


torch.backends.cudnn.benchmark = True
fit(
    model,
    opt,
    loss_fn,
    epochs=50,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['categorical_accuracy']
)
