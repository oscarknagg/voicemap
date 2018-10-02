import numpy as np
from keras.callbacks import Callback
from keras.models import clone_model
from tqdm import tqdm
from scipy.spatial.distance import cdist
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


def n_shot_task_evaluation(model, dataset, preprocessor, num_tasks, n, k, network_type='siamese', distance='euclidean'):
    """Evaluate a siamese network on k-way, n-shot classification tasks generated from a particular dataset.

    # Arguments
        model: Model to evaluate
        dataset: Dataset (currently LibriSpeechDataset only) from which to build evaluation tasks
        preprocessor: Preprocessing function to apply to samples
        num_tasks: Number of tasks to evaluate with
        n: Number of samples per class present in the support set
        k: Number of classes present in the support set
        network_type: Either 'siamese' or 'classifier'. This controls how to get the embedding function from the model
        distance: Either 'euclidean' or 'cosine'. This controls how to combine the support set samples for n > 1 shot
        tasks
    """
    # TODO: Faster/multiprocessing creation of tasks
    n_correct = 0

    if n == 1 and network_type == 'siamese':
        # Directly use siamese network to get pairwise verficiation score, minimum is closest
        for i_eval in tqdm(range(num_tasks)):
            query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

            input_1 = np.stack([query_sample[0]] * k)[:, :, np.newaxis]
            input_2 = support_set_samples[0][:, :, np.newaxis]

            # Perform preprocessing
            # Pass an empty list to the labels parameter as preprocessor functions on batches not samples
            ([input_1, input_2], _) = preprocessor(([input_1, input_2], []))

            pred = model.predict([input_1, input_2])

            if np.argmin(pred[:, 0]) == 0:
                # 0 is the correct result as by the function definition
                n_correct += 1
    elif n > 1 or network_type == 'classifier':
        # Create encoder network from earlier layers
        if network_type == 'siamese':
           encoder = model.layers[2]
        elif network_type == 'classifier':
            encoder = clone_model(model)
            encoder.set_weights(model.get_weights())
            encoder.pop()
        else:
            raise(ValueError, 'mode must be one of (siamese, classifier)')

        for i_eval in tqdm((range(num_tasks))):
            query_sample, support_set_samples = dataset.build_n_shot_task(k, n)

            # Perform preprocessing
            query_instance = preprocessor.instance_preprocessor(query_sample[0].reshape(1, -1, 1))
            support_set_instances = preprocessor.instance_preprocessor(support_set_samples[0][:, :, np.newaxis])

            query_embedding = encoder.predict(query_instance)
            support_set_embeddings = encoder.predict(support_set_instances)

            if distance == 'euclidean':
                # Get mean position of support set embeddings
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                # TODO: replace for loop with np.ufunc.reduceat
                mean_support_set_embeddings = []
                for i in range(0, n * k, n):
                    mean_support_set_embeddings.append(support_set_embeddings[i:i + n, :].mean(axis=0))
                mean_support_set_embeddings = np.stack(mean_support_set_embeddings)

                # Get euclidean distances between mean embeddings
                pred = np.sqrt(
                    np.power((np.concatenate([query_embedding] * k) - mean_support_set_embeddings), 2).sum(axis=1))
            elif distance == 'cosine':
                # Get "mean" position of support set embeddings. Do this by calculating the per-class mean angle
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                magnitudes = np.linalg.norm(support_set_embeddings, axis=1, keepdims=True)
                unit_vectors = support_set_embeddings / magnitudes
                mean_support_set_unit_vectors = []
                for i in range(0, n * k, n):
                    mean_support_set_unit_vectors.append(unit_vectors[i:i + n, :].mean(axis=0))
                    # mean_support_set_magnitudes.append(magnitudes[i:i + n].sum() / n)

                mean_support_set_unit_vectors = np.stack(mean_support_set_unit_vectors)

                # Get cosine distance between angular-mean embeddings
                pred = cdist(query_embedding, mean_support_set_unit_vectors, 'cosine')
            elif distance == 'dot_product':
                # Get "mean" position of support set embeddings. Do this by calculating the per-class mean angle and
                # magnitude.
                # This is very similar to 'cosine' except in the case that two support set samples have the same angle,
                # in which case the one with the larger magnitude will be preffered
                # Assumes a label structure like [class_1]*n + [class_2]*n + ... + [class_k]*n
                magnitudes = np.linalg.norm(support_set_embeddings, axis=1, keepdims=True)
                unit_vectors = support_set_embeddings / magnitudes
                mean_support_set_unit_vectors = []
                mean_support_set_magnitudes = []
                for i in range(0, n * k, n):
                    mean_support_set_unit_vectors.append(unit_vectors[i:i + n, :].mean(axis=0))
                    mean_support_set_magnitudes.append(magnitudes[i:i + n].sum() / n)

                mean_support_set_unit_vectors = np.stack(mean_support_set_unit_vectors)
                mean_support_set_magnitudes = np.vstack(mean_support_set_magnitudes)
                mean_support_set_embeddings = mean_support_set_magnitudes * mean_support_set_unit_vectors

                # Get dot product between mean embeddings
                pred = np.dot(query_embedding[0, :][np.newaxis, :], mean_support_set_embeddings.T)
                # As dot product is a kind of similarity let's make this a "distance" by flipping the sign
                pred = -pred
            else:
                raise(ValueError, 'Distance must be in (euclidean, cosine, dot_product)')

            if np.argmin(pred) == 0:
                # 0 is the correct result as by the function definition
                n_correct += 1
    else:
        raise(ValueError, "n must be >= 1")

    return n_correct


class NShotEvaluationCallback(Callback):
    """Evaluate a siamese network on n-shot classification tasks after every epoch.

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
        # self.evaluator = evaluate_siamese_network if mode == 'siamese' else evaluate_classification_network

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        n_correct = n_shot_task_evaluation(self.model, self.dataset, self.preprocessor, self.num_tasks, self.n_shot,
                                           self.k_way, network_type=self.mode)

        n_shot_acc = n_correct * 1. / self.num_tasks
        logs['val_{}-shot_acc'.format(self.n_shot)] = n_shot_acc

        print 'val_{}-shot_acc: {:.4f}'.format(self.n_shot, n_shot_acc)
