from torch import nn
from torch.utils.data import DataLoader

from voicemap.callbacks import DefaultCallback, ProgbarLogger, CallbackList


def gradient_step(model, optimiser, loss_fn, batch, **kwargs):
    """Takes a single gradient step.

    TODO: Accumulent gradients for arbitrary effective batch size
    """
    model.train()
    optimiser.zero_grad()
    x, y = batch
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss.item()


def fit(model: nn.Module, optimiser, loss_fn, epochs: int, dataloader: DataLoader, prepare_batch, callbacks=None,
        verbose=True):
    # Determine number of samples:
    num_batches = len(dataloader)

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgbarLogger(num_batches), ])

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = dataloader.batch_size

            callbacks.on_batch_begin(batch_index, batch_logs)

            batch = prepare_batch(batch)

            loss = gradient_step(model, optimiser, loss_fn, batch)
            batch_logs['loss'] = loss

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
            callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
