import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from voicemap.librispeech import LibriSpeechDataset
from voicemap.models import get_classifier
from voicemap.callbacks import CSVLogger
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')


##############
# Parameters #
##############
filters = 128
embedding = 64
batchsize = 96
n_seconds = 3
downsampling = 4


###################
# Create datasets #
###################
dataset = ['train-clean-100']
data = LibriSpeechDataset(dataset, n_seconds, downsampling, stochastic=False)

indices = range(len(data))
train_indices, test_indices, _, _ = train_test_split(indices, indices, test_size=0.97)

train = torch.utils.data.Subset(data, train_indices)
test = torch.utils.data.Subset(data, test_indices)

print(len(train), len(test))


################
# Define model #
################
model = get_classifier(filters, embedding, data.num_classes())
model.to(device, dtype=torch.double)


############
# Training #
############
from voicemap.train import fit


train_loader = DataLoader(train, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def prepare_batch(batch):
    # Move to GPU and convert targets to int
    x, y = batch
    return x.cuda(), y.long().cuda()


callbacks = [CSVLogger(PATH + '/logs/pytorch.csv')]

torch.backends.cudnn.benchmark = True
fit(
    model,
    opt,
    loss_fn,
    epochs=5,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks
)
