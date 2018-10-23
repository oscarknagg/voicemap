"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from voicemap.datasets import OmniglotDataset, MiniImageNet
from voicemap.models import get_omniglot_classifier, Bottleneck
from voicemap.few_shot import NShotWrapper
from voicemap.train import fit
from voicemap.callbacks import *
from voicemap.metrics import categorical_accuracy
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
dataset = 'omniglot'

if dataset == 'omniglot':
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
else:
    n_shot_train = 1
    k_way_train = 60
    q_queries_train = 5
    n_epochs = 40
    episodes_per_epoch = 100
    n_shot_val = 1
    k_way_val = 5
    q_queries_val = 1
    evaluation_episodes = 1000
    dataset_class = MiniImageNet


####################
# Helper functions #
####################
def prepare_nshot_task(n, k, q):
    def prepare_nshot_task_(batch):
        # Strip extra batch dimension from inputs and outputs
        # The extra batch dimension is a consequence of using the DataLoader
        # class. However the DataLoader gives easy multiprocessing
        x, y = batch
        x = x.reshape(x.shape[1:]).cuda()
        # Create dummy 0-(num_classes - 1) label
        y = torch.arange(0, k, 1 / q).long().cuda()
        return x, y

    return prepare_nshot_task_


def proto_net_episode(model, optimiser, loss_fn, x, y, **kwargs):
    if kwargs['train']:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

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

    if kwargs['train']:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss.item(), y_pred


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % 20 == 0:
        return lr / 2
    else:
        return lr


class EvaluateProtoNet(Callback):
    """Evaluate a prototypical network network on n-shot, k-way classification tasks after every epoch.

        # Arguments
            num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            task_loader: Instance of NShotWrapper class
            prepare_batch: function. The preprocessing function to apply to samples from the dataset.
            prefix: str. Prefix to identify dataset.
        """

    def __init__(self, num_tasks, n_shot, k_way, q_queries, task_loader, prepare_batch, prefix='val_'):
        super(EvaluateProtoNet, self).__init__()
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = task_loader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = proto_net_episode(self.model, self.optimiser, self.loss_fn, x, y,
                                             n_shot=self.n_shot, k_way=self.k_way, q_queries=self.q_queries,
                                             train=False)

            seen += y_pred.shape[0]

            totals['loss'] += loss * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen


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
# This creates the baseline Omniglot classifier and then strips the classification layer leaving just
# a network that embeds characters into a 64D space.
model = Bottleneck(get_omniglot_classifier(1))
model.to(device, dtype=torch.double)


############
# Training #
############
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()

callbacks = [
    EvaluateProtoNet(num_tasks=evaluation_episodes, n_shot=n_shot_val, k_way=k_way_val, q_queries=q_queries_val,
                     task_loader=evaluation_taskloader,
                     prepare_batch=prepare_nshot_task(n_shot_val, k_way_val, q_queries_val)),
    ModelCheckpoint(filepath=PATH + '/models/proto_net_omniglot.torch', monitor='val_1-shot_5-way_acc'),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + '/logs/proto_net_omniglot.csv'),
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
