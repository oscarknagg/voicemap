"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import torch
import numpy as np
from torch.optim import Adam

from voicemap.datasets import OmniglotDataset
from voicemap.models import get_omniglot_classifier, Bottleneck
from voicemap.eval import n_shot_k_way_evaluation
from voicemap.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')


##############
# Parameters #
##############
batchsize = 64
test_fraction = 0.1
num_tasks = 100
k_way = 60
n_shot = 1
query_samples_per_class = 5
n_epochs = 40
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


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % 20 == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    NShotTaskEvaluation(num_tasks=num_tasks, n_shot=n_shot, k_way=k_way, dataset=evaluation,
                        prepare_batch=prepare_n_shot_batch, prefix='val_', network_type='encoder'),
    ModelCheckpoint(filepath=PATH + '/models/proto_net.torch', monitor='val__1-shot_5-way_acc'),
    LearningRateScheduler(schedule=lr_schedule, verbose=True),
    CSVLogger(PATH + '/logs/proto_net.csv'),
]
callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
callbacks.set_model(model)
callbacks.set_params({
    'metrics': [],
    'verbose': True,
    'optimiser': optimiser,
    'num_batches': episodes_per_epoch
})
callbacks.on_train_begin()


# Implement Algorithm 1 from Prototypical Networks
for epoch in range(1, n_epochs+1):
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    for episode in range(episodes_per_epoch):
        batch_logs = dict(batch=episode, size=1)
        callbacks.on_batch_begin(episode, batch_logs)

        # Initialise training for episode
        model.train()
        optimiser.zero_grad()
        loss = 0

        # Select classes
        episode_classes = np.random.choice(background.df['class_id'].unique(), size=k_way, replace=False)
        df = background.df[background.df['class_id'].isin(episode_classes)]

        prototypes = {k: None for k in episode_classes}
        queries = {k: None for k in episode_classes}
        for k in episode_classes:
            # Select support examples
            support = df[df['class_id'] == k].sample(n_shot)

            # Create prototype
            embeddings = []
            for i, s in support.iterrows():
                x, y = background[s['id']]
                x = x[np.newaxis, :, :, :]
                x = torch.from_numpy(x).cuda()
                embeddings.append(
                    model(x)
                )

            prototype = torch.cat(embeddings)
            prototype = prototype.mean(dim=0, keepdim=True)
            prototypes[k] = prototype

            # Select query samples
            query = df[(df['class_id'] == k) & (~df['id'].isin(support['id']))].sample(query_samples_per_class)
            class_queries = []
            for i, q in query.iterrows():
                x, y = background[q['id']]
                x = x[np.newaxis, :, :, :]
                x = torch.from_numpy(x).cuda()
                class_queries.append(model(x))

            queries[k] = class_queries

        # Calculate loss
        for k in episode_classes:
            q_k = queries[k]
            for x in q_k:
                # The code below implements the folling equation from Algorithm 1
                # J <- J + (1/N_c * N_q) [d(f_phi(x), c_k) + log \Sigma_k exp( - d(f_phi(x), c_k) ) ]
                loss += scaling_factor * (
                    # d(f_phi(x), c_k)
                    torch.pairwise_distance(x, prototypes[k]) +
                    # log
                    torch.log(
                        # Sum over classes k_prime
                        torch.cat([
                            # Negative exponential distance exp( - d(f_phi(x), c_k) )
                            torch.exp(-torch.pairwise_distance(x, proto))
                            for proto in prototypes.values()]
                        ).sum()
                    )
                )

        # Convert loss to scalar (O-dim Tensor) from size (1,) Tensor
        loss.backward()
        optimiser.step()

        batch_logs['loss'] = loss.item()
        callbacks.on_batch_end(episode, batch_logs)

    callbacks.on_epoch_end(epoch, epoch_logs)

callbacks.on_train_end()
