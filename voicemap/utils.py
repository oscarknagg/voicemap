import numpy as np
from keras.callbacks import Callback
from tqdm import tqdm
import keras.backend as K


def get_bottleneck(classifier, samples):
    """Ripped from https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer"""
    inp = classifier.input  # input placeholder
    outputs = [layer.output for layer in classifier.layers]  # all layer outputs
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

    # Get activations
    layer_outs = functor([samples, 0.])

    # Return bottleneck only
    return layer_outs[-2]


def preprocess_instances(downsampling, whitening=True):
    """This is the canonical preprocessing function for this project.

    1. Downsampling audio segments to desired sampling rate
    2. Whiten audio segments to 0 mean and fixed RMS (aka volume)
    """
    def preprocess_instances_(instances):
        instances = instances[:, ::downsampling, :]
        if whitening:
            instances = whiten(instances)
        return instances

    return preprocess_instances_


class BatchPreProcessor(object):
    """Wrapper class for instance and label pre-processing.

    This class implements a __call__ method that pre-process classifier-style batches (inputs, outputs) and siamese
    network-style batches ([input_1, input_2], outputs) identically.

    # Arguments
        mode: str. One of {siamese, classifier)
        instance_preprocessor: function. Pre-processing function to apply to input features of the batch.
        target_preprocessor: function. Pre-processing function to apply to output labels of the batch.
    """
    def __init__(self, mode, instance_preprocessor, target_preprocessor=lambda x: x):
        assert mode in ('siamese', 'classifier')
        self.mode = mode
        self.instance_preprocessor = instance_preprocessor
        self.target_preprocessor = target_preprocessor

    def __call__(self, batch):
        """Pre-processes a batch of samples."""
        if self.mode == 'siamese':
            ([input_1, input_2], labels) = batch

            input_1 = self.instance_preprocessor(input_1)
            input_2 = self.instance_preprocessor(input_2)

            labels = self.target_preprocessor(labels)

            return [input_1, input_2], labels
        elif self.mode == 'classifier':
            instances, labels = batch

            instances = self.instance_preprocessor(instances)

            labels = self.target_preprocessor(labels)

            return instances, labels
        else:
            raise ValueError


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(
        (1 - y_true) * K.square(y_pred) +
        y_true * K.square(K.maximum(margin - y_pred, 0))
    )


def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (n_segments, n_timesteps, 1).')

    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    whitened_batch = batch - np.tile(sample_wise_mean, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    # Divide through
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    return whitened_batch


def evaluate_siamese_network(siamese, dataset, preprocessor, num_tasks, n, k):
    """Evaluate a siamese network on k-way, n-shot classification tasks generated from a particular dataset."""
    # Currently assumes 1 shot classification in evaluation task
    if n != 1:
        raise NotImplementedError

    # TODO: Faster/multiprocessing creation of tasks
    n_correct = 0
    for i_eval in tqdm(range(num_tasks)):
        query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

        input_1 = np.stack([query_sample[0]] * k)[:, :, np.newaxis]
        input_2 = support_set_samples[0][:, :, np.newaxis]

        # Perform preprocessing
        # Pass an empty list to the labels parameter as preprocessor functions on batches not samples
        ([input_1, input_2], _) = preprocessor(([input_1, input_2], []))

        pred = siamese.predict([input_1, input_2])

        if np.argmin(pred[:, 0]) == 0:
            # 0 is the correct result as by the function definition
            n_correct += 1

    return n_correct


def evaluate_siamese_network_nshot(siamese, dataset, preprocessor, num_tasks, n, k):
    encoder = siamese.layers[2]
    encoder.build()

    n_correct = 0
    for i_eval in tqdm(range(num_tasks)):
        query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

        # Perform preprocessing
        query_instance = preprocessor.instance_preprocessor(query_sample[0])
        support_set_instances = preprocessor.instance_preprocessor(support_set_samples[0])

        query_embedding = encoder.predict(query_instance)
        support_set_embeddings = encoder.predict(support_set_instances)

        # Get mean position of support set embeddings
        # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k] * n
        # TODO: write a test for this
        # TODO: replace for loop with np.ufunc.reduceat
        mean_support_set_embeddings = []
        for i in range(0, n*k, n):
            mean_support_set_embeddings.append(support_set_embeddings[i:i+n, :].mean(axis=1))
        mean_support_set_embeddings = np.stack(mean_support_set_embeddings)

        # Get euclidean distances between mean embeddings
        pred = np.sqrt(np.power((np.concatenate([query_embedding] * k) - mean_support_set_embeddings), 2).sum(axis=1))

        if np.argmin(pred) == 0:
            # 0 is the correct result as by the function definition
            n_correct += 1

    return n_correct


def evaluate_classification_network(model, dataset, preprocessor, num_tasks, n, k):
    """Evaluate a classification network on  k-way, n-shot classification tasks generated from a particular dataset.

    We will use euclidean distances between the activations of the penultimate "bottleneck" layer for each sample as the
    similarity metric.
    """
    if n != 1:
        raise NotImplementedError

    n_correct = 0
    for i_eval in tqdm(range(num_tasks)):
        query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

        query_sample = (np.array([query_sample[0]])[:, :, np.newaxis], np.array([query_sample[1]])[:, np.newaxis])
        support_set_samples = (
            support_set_samples[0][:, :, np.newaxis],
            support_set_samples[1][:, np.newaxis]
        )

        # Perform preprocessing
        query_instance = preprocessor.instance_preprocessor(query_sample[0])
        support_set_instances = preprocessor.instance_preprocessor(support_set_samples[0])

        # Get bottleneck activations for query and support set
        query_embedding = get_bottleneck(model, query_instance)
        support_set_embeddings = get_bottleneck(model, support_set_instances)

        # Get euclidean distances between embeddings
        pred = np.sqrt(np.power((np.concatenate([query_embedding] * k) - support_set_embeddings), 2).sum(axis=1))

        if np.argmin(pred) == 0:
            # 0 is the correct result as by the function definition
            n_correct += 1

    return n_correct


class NShotEvaluationCallback(Callback):
    """Evaluate a siamese network on n-shot classification tasks after every epoch.

    Can also optionally log various metrics to CSV and save best model according to n-shot classification accuracy.

    # Arguments
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        dataset: LibriSpeechDataset. The dataset to generate the n-shot classification tasks from.
        preprocessor: function. The preprocessing function to apply to samples from the dataset.
        verbose: bool. Whether to enable verbose printing
        mode: str. One of {siamese, classifier}
    """
    def __init__(self, num_tasks, n_shot, k_way, dataset, preprocessor=lambda x: x, mode='siamese'):
        super(NShotEvaluationCallback, self).__init__()
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.dataset = dataset
        self.preprocessor = preprocessor

        assert mode in ('siamese', 'classifier')
        self.mode = mode
        self.evaluator = evaluate_siamese_network if mode == 'siamese' else evaluate_classification_network

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        n_correct = self.evaluator(self.model, self.dataset, self.preprocessor, self.num_tasks, self.n_shot, self.k_way)

        n_shot_acc = n_correct * 1. / self.num_tasks
        logs['val_{}-shot_acc'.format(self.n_shot)] = n_shot_acc

        print 'val_{}-shot_acc: {:.4f}'.format(self.n_shot, n_shot_acc)
