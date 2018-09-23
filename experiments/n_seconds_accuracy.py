"""
This experiment tests the accuracy of models trained on different length audio fragments.
"""
import os
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import multiprocessing

from voicemap.utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
from voicemap.models import get_baseline_convolutional_encoder, build_siamese_net
from voicemap.librispeech import LibriSpeechDataset
from config import LIBRISPEECH_SAMPLING_RATE, PATH


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
n_repeats = 1
n_seconds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
downsampling = 4
batchsize = 32
model_n_filters = 128
model_embedding_dimension = 64
model_dropout = 0.0
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
num_epochs = 50
evaluate_every_n_batches = 1000
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5


#################
# Training Loop #
#################
for fragment_length in n_seconds:
    print '*' * 23
    print '***** {:.1f} seconds *****'.format(fragment_length)
    print '*' * 23
    input_length = int(LIBRISPEECH_SAMPLING_RATE * fragment_length / downsampling)

    # Create datasets
    train = LibriSpeechDataset(training_set, fragment_length, pad=True)
    valid = LibriSpeechDataset(validation_set, fragment_length, stochastic=False, pad=True)

    batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
    train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))
    valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))

    for repeat in range(n_repeats):
        # Define model
        encoder = get_baseline_convolutional_encoder(model_n_filters, model_embedding_dimension, dropout=model_dropout)
        siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
        opt = Adam(clipnorm=1.)
        siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Train
        param_str = 'siamese__nseconds_{}__filters_{}__embed_{}__drop_{}__r_{}'.format(fragment_length, model_n_filters,
                                                                                       model_embedding_dimension,
                                                                                       model_dropout, repeat)
        print param_str
        siamese.fit_generator(
            generator=train_generator,
            steps_per_epoch=evaluate_every_n_batches,
            validation_data=valid_generator,
            validation_steps=100,
            epochs=num_epochs,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True,
            callbacks=[
                # First generate custom n-shot classification metric
                NShotEvaluationCallback(
                    num_evaluation_tasks, n_shot_classification, k_way_classification, valid,
                    preprocessor=batch_preprocessor,
                ),
                # Then log and checkpoint
                CSVLogger(PATH + '/logs/n_seconds/{}.csv'.format(param_str)),
                ModelCheckpoint(
                    PATH + '/models/n_seconds/{}.hdf5'.format(param_str),
                    monitor='val_{}-shot_acc'.format(n_shot_classification),
                    mode='max',
                    save_best_only=True,
                    verbose=True
                ),
                ReduceLROnPlateau(
                    monitor='val_{}-shot_acc'.format(n_shot_classification),
                    mode='max',
                    verbose=1
                )
            ]
        )
