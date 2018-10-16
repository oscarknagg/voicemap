import torch

from voicemap.models import Bottleneck
from voicemap.metrics import NAMED_METRICS


def evaluate(model, dataloader, prepare_batch, metrics, loss_fn=None, prefix='val_', suffix=''):
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs


def n_shot_k_way_evaluation(model, dataset, prepare_batch, num_tasks, n, k,
                            network_type='classifier', distance='euclidean'):
    if network_type != 'classifier':
        raise NotImplementedError

    if distance != 'euclidean':
        raise NotImplementedError

    if n > 1:
        raise NotImplementedError

    bottleneck = Bottleneck(model)
    bottleneck.eval()

    n_correct = 0
    for i in range(num_tasks):
        query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

        query_instance, support_instances = prepare_batch(query_sample, support_set_samples)

        with torch.no_grad():
            query_embedding = bottleneck(query_instance)
            support_embeddings = bottleneck(support_instances)

        pred = torch.pairwise_distance(
            query_embedding.repeat([support_embeddings.shape[0], 1]),
            support_embeddings,
            keepdim=True
        )

        n_correct += (pred.argmin() == 0).item()

    return n_correct / num_tasks
