import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from multiprocessing import cpu_count
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from olympic.callbacks import Callback
from tqdm import tqdm

from voicemap.datasets import PairDataset, collate_pairs


def seg_intersect(a1,a2, b1,b2):
    def perp(a) :
        # Gets a gradient perpdendicular to a
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap,db)
    num = np.dot(dap,dp)
    return (num / denom.astype(float))*db + b1


def equal_error_rate(fpr, tpr):
    metrics = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    metrics['diff'] = metrics['tpr'] - (1 - metrics['fpr'])
    upper = metrics[metrics['diff'] > 0].sort_values('diff').head(1)
    lower = metrics[metrics['diff'] < 0].sort_values('diff', ascending=False).head(1)
    # Line between
    a1 = np.array([lower['fpr'].values[0], (1 - lower['tpr']).values[0]])
    a2 = np.array([upper['fpr'].values[0], (1 - upper['tpr']).values[0]])
    # Line for equal error rates
    b1 = np.array([0, 0])
    b2 = np.array([1, 1])
    return seg_intersect(a1, a2, b1, b2)[0]


def unseen_speakers_evaluation(model: nn.Module, dataset: Dataset, num_pairs: int) -> dict:
    """Calculates AUC and EER for verification on unseen speakers"""
    unseen_speaker_pairs = PairDataset(dataset, num_pairs=num_pairs)
    loader = DataLoader(unseen_speaker_pairs, batch_size=100, collate_fn=collate_pairs, num_workers=cpu_count(),
                        drop_last=False)

    distances = []
    labels = []

    pbar = tqdm(total=len(loader))
    model.eval()
    for i, ((x, y), label) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x_embed = model(x, return_embedding=True)
            y_embed = model(y, return_embedding=True)

            sim = F.cosine_similarity(x_embed, y_embed, dim=1, eps=1e-6)
            distances += sim.tolist()
            labels += label

        pbar.update(1)

    pbar.close()

    roc_auc = roc_auc_score(labels, distances)
    fpr, tpr, _ = roc_curve(labels, distances)
    eer = equal_error_rate(fpr, tpr)
    return {
        'auc': roc_auc,
        'eer': eer,
    }


class VerificationMetrics(Callback):
    def __init__(self, dataset: Dataset, num_pairs: int, prefix: str = 'val_', suffix: str = ''):
        super(VerificationMetrics, self).__init__()
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.prefix = prefix
        self.suffix = suffix

    def on_train_begin(self, logs=None):
        self.metrics = self.params['metrics']
        self.prepare_batch = self.params['prepare_batch']
        self.loss_fn = self.params['loss_fn']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({
            self.prefix+k+self.suffix: v
            for k, v in unseen_speakers_evaluation(self.model, self.dataset, self.num_pairs).items()
        })
