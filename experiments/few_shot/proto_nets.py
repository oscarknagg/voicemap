"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from voicemap.datasets import OmniglotDataset, MiniImageNet
from voicemap.models import get_few_shot_encoder
from voicemap.few_shot import NShotWrapper, proto_net_episode, EvaluateProtoNet, prepare_nshot_task
from voicemap.train import fit
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
elif args.dataset == 'miniImageNet':
    n_shot_train = 1
    k_way_train = 30
    q_queries_train = 15
    n_epochs = 80
    episodes_per_epoch = 100
    n_shot_val = 1
    k_way_val = 5
    q_queries_val = 1
    evaluation_episodes = 1000
    dataset_class = MiniImageNet
    num_input_channels = 3
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'proto_net_{args.dataset}_n={n_shot_train}_k={k_way_train}_q={q_queries_train}'


###################
# Create datasets #
###################
background = dataset_class('background')
background_tasks = NShotWrapper(background, episodes_per_epoch, n_shot_train, k_way_train, q_queries_train)
background_taskloader = DataLoader(background_tasks, batch_size=1, num_workers=4)
evaluation = dataset_class('evaluation')
evaluation_tasks = NShotWrapper(evaluation, evaluation_episodes, n_shot_val, k_way_val, q_queries_val)
evaluation_taskloader = DataLoader(evaluation_tasks, batch_size=1, num_workers=4)


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)


############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % 20 == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateProtoNet(
        num_tasks=evaluation_episodes,
        n_shot=n_shot_val,
        k_way=k_way_val,
        q_queries=q_queries_val,
        task_loader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(n_shot_val, k_way_val, q_queries_val)
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/{param_str}.torch',
        monitor=f'val_{n_shot_val}-shot_{k_way_val}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(n_shot_train, k_way_train, q_queries_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': n_shot_train, 'k_way': k_way_train, 'q_queries': q_queries_train, 'train': True}
)
