from torch import nn
from torch.utils.data import DataLoader

from voicemap.callbacks import DefaultCallback, ProgressBarLogger, CallbackList
from voicemap.metrics import NAMED_METRICS


def gradient_step(model, optimiser, loss_fn, x, y):
    """Takes a single gradient step.

    TODO: Accumulent gradients for arbitrary effective batch size
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss.item(), y_pred


def fit(model, optimiser, loss_fn, epochs: int, dataloader, prepare_batch, metrics=None, callbacks=None, verbose=True):
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=batch_size)

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)

            loss, y_pred = gradient_step(model, optimiser, loss_fn, x, y)
            batch_logs['loss'] = loss

            model.eval()
            for m in metrics:
                batch_logs[m] = NAMED_METRICS[m](y, y_pred)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
