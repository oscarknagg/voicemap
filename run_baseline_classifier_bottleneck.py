import os
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils import to_categorical, Sequence
import numpy as np
import multiprocessing

from utils import BatchPreProcessor, preprocess_instances, NShotEvaluationCallback
from models import get_baseline_convolutional_encoder
from librispeech import LibriSpeechDataset
from config import LIBRISPEECH_SAMPLING_RATE, PATH


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
model_n_filters = 32
model_embedding_dimension = 128
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
num_epochs = 25
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

# Derived parameters
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)


###################
# Create datasets #
###################
train = LibriSpeechDataset(training_set, n_seconds)
valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False)

# Map speaker IDs to the range 0 - (train.num_classes() - 1)
unique_speakers = sorted(train.df['speaker_id'].unique())
speaker_id_mapping = {unique_speakers[i]: i for i in range(train.num_classes())}


class BatchedSequence(Sequence):
    """Convenience class """
    def __init__(self, sequence, batch_preprocessor, batchsize):
        self.sequence = sequence
        self.batch_preprocessor = batch_preprocessor
        self.batchsize = batchsize

        # Initialise index to batch mapping
        self.underlying_indexes = range(len(sequence))

        np.random.shuffle(self.underlying_indexes)
        self.batch_to_index = {i: self.underlying_indexes[i*batchsize:(i+1)*batchsize] for i in range(len(self))}

    def __len__(self):
        return int(len(self.sequence) / float(self.batchsize))

    def __getitem__(self, item):
        # Get batches from underlying Sequence
        X = []
        y = []
        for underlying_i in self.batch_to_index[item]:
            sample = self.sequence[underlying_i]
            X.append(sample[0][:, np.newaxis])
            y.append(sample[1])

        X = np.stack(X)
        y = np.stack(y)[:, np.newaxis]

        # Preprocess
        X, y = self.batch_preprocessor((X, y))

        return X, y

    def on_epoch_end(self):
        # Shuffle the indexes
        np.random.shuffle(self.underlying_indexes)
        self.batch_to_index = {i: self.underlying_indexes[i * batchsize:(i + 1) * batchsize] for i in range(len(self))}


def label_preprocessor(num_classes, speaker_id_mapping):
    def label_preprocessor_(y):
        y = np.array([speaker_id_mapping[i] for i in y[:, 0]])[:, np.newaxis]
        return to_categorical(y, num_classes)

    return label_preprocessor_


batch_preprocessor = BatchPreProcessor('classifier', preprocess_instances(downsampling),
                                       label_preprocessor(train.num_classes(), speaker_id_mapping))

train_generator = BatchedSequence(train, batch_preprocessor, batchsize)


################
# Define model #
################
classifier = get_baseline_convolutional_encoder(model_n_filters, model_embedding_dimension, (input_length, 1))
# Add output classification layer
classifier.add(Dense(train.num_classes(), activation='softmax'))

opt = Adam(clipnorm=1.)
classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
plot_model(classifier, show_shapes=True, to_file=PATH + '/plots/classifier.png')
print classifier.summary()


#################
# Training Loop #
#################
classifier.fit_generator(
    generator=train_generator,
    steps_per_epoch=evaluate_every_n_batches,
    epochs=num_epochs,
    workers=multiprocessing.cpu_count(),
    use_multiprocessing=True,
    callbacks=[
        # First generate custom n-shot classification metric
        NShotEvaluationCallback(
            num_evaluation_tasks, n_shot_classification, k_way_classification, valid,
            preprocessor=batch_preprocessor, mode='classifier'
        ),
        # # Then log and checkpoint
        # CSVLogger(PATH + '/logs/baseline_classifier.csv'),
        # ModelCheckpoint(
        #     PATH + '/models/baseline_classifier.hdf5',
        #     monitor='val_{}-shot_acc'.format(n_shot_classification),
        #     mode='max',
        #     save_best_only=True,
        #     verbose=True
        # )
    ]
)