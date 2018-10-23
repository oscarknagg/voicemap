"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from voicemap.datasets import OmniglotDataset
from voicemap.models import get_omniglot_classifier, Bottleneck
from voicemap.eval import n_shot_k_way_evaluation
from voicemap.train import fit
from voicemap.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
batchsize = 64
test_fraction = 0.1
num_tasks = 1000
k_way = 60
n_shot = 1
query_samples_per_class = 2
n_epochs = 60
episodes_per_epoch = 100

scaling_factor = (1 / (k_way * query_samples_per_class))


####################
# Helper functions #
####################
def prepare_n_shot_batch(query, support):
    query = torch.from_numpy(query[0]).to(device, dtype=torch.double)
    support = torch.from_numpy(support[0]).to(device, dtype=torch.double)
    return query, support


###################
# Create datasets #
###################
background = OmniglotDataset('background')
evaluation = OmniglotDataset('evaluation')


#########
# Model #
#########
# This creates the baseline Omniglot classifier and then strips the classification layer leaving just
# a network that embeds characters into a 64D space.
model = Bottleneck(get_omniglot_classifier(1))
model.to(device, dtype=torch.double)


############
# Training #
############
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


class NShotWrapper(Dataset):
    def __init__(self, dataset, epoch_length, n, k, q):
        self.dataset = dataset
        self.epoch_length = epoch_length
        self.n_shot = n
        self.k_way = k
        self.q_queries = q

    def __getitem__(self, item):
        """Get a single n-shot, k-way, q-query task."""
        # Select classes
        episode_classes = np.random.choice(background.df['class_id'].unique(), size=self.k_way, replace=False)
        df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
        batch = []

        for k in episode_classes:
            # Select support examples
            support = df[df['class_id'] == k].sample(self.n_shot)

            for i, s in support.iterrows():
                x, y = self.dataset[s['id']]
                batch.append(x)

        for k in episode_classes:
            query = df[(df['class_id'] == k) & (~df['id'].isin(support['id']))].sample(self.q_queries)
            for i, q in query.iterrows():
                x, y = self.dataset[q['id']]
                batch.append(x)

        return np.stack(batch), episode_classes

    def __len__(self):
        return self.epoch_length


background_tasks = NShotWrapper(background, episodes_per_epoch, n_shot, k_way, query_samples_per_class)
background_taskloader = DataLoader(background_tasks, batch_size=1, num_workers=4)


def prepare_batch(batch):
    # Strip extra batch dimension from inputs and outputs
    # The extra batch dimension is a consequence of using the DataLoader
    # class. However the DataLoader gives easy multiprocessing
    x, y = batch
    x = x.reshape(x.shape[1:]).cuda()
    return x, torch.arange(0, k_way, 1/query_samples_per_class).long().cuda()


def gradient_step(model, optimiser, loss_fn, x, y, **kwargs):
    # Zero gradients
    model.train()
    optimiser.zero_grad()

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:kwargs['n_shot']*kwargs['k_way']]
    queries = embeddings[kwargs['n_shot']*kwargs['k_way']:]

    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    prototypes = support.reshape(kwargs['n_shot'], kwargs['k_way'], -1).mean(dim=0)

    # Efficiently calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way)
    distances = (
        queries.unsqueeze(1).expand(kwargs['q_queries'] * kwargs['k_way'], kwargs['k_way'], -1) -
        prototypes.unsqueeze(0).expand(kwargs['q_queries'] * kwargs['k_way'], kwargs['k_way'], -1)
    ).pow(2).sum(dim=2)
    # print(distances)
    logits = -distances

    # First instance is always correct one by construction so the label reflects this
    # Label is repeated by the number of queries
    loss = loss_fn(logits, y)

    # Prediction probabilities are softmax over distances
    y_pred = logits.softmax(dim=1)

    # Take gradient step
    loss.backward()
    optimiser.step()

    return loss.item(), y_pred


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % 20 == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    NShotTaskEvaluation(num_tasks=num_tasks, n_shot=1, k_way=5, dataset=evaluation,
                        prepare_batch=prepare_n_shot_batch, prefix='val_', network_type='encoder'),
    # ModelCheckpoint(filepath=PATH + '/models/proto_net_omniglot.torch', monitor='val_1-shot_5-way_acc'),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + '/logs/proto_net_omniglot.csv'),
]


fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=gradient_step,
    fit_function_kwargs={'n_shot': n_shot, 'k_way': k_way, 'q_queries': query_samples_per_class}
)
