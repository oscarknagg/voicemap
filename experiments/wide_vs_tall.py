"""
This experiment tests the accuracy of models trained on different length audio fragments.
"""
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import multiprocessing

from voicemap.utils import preprocess_instances, NShotEvaluationCallback, BatchPreProcessor
from voicemap.librispeech import LibriSpeechDataset
from voicemap.models import get_baseline_convolutional_encoder, build_siamese_net
from config import LIBRISPEECH_SAMPLING_RATE, PATH


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
model_n_filters = 128
model_embedding_dimension = 64
model_dropout = 0.0
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
num_epochs = 50
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5
pad = True

min_speakers = 32
max_speakers = 800
max_minutes = 25
n_repeats = 1
n_points = 10  # Number of wide vs tall points to sample

# Derived parameters
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)


####################
# Helper functions #
####################
def create_reduced_dataset(dataset, n_speakers, seconds_per_speaker):
    # Copy the dataset class and modify it's internal dataframe
    new_dataset = deepcopy(dataset)
    df = new_dataset.df

    # Take a random sample of speakers with at least `seconds_per_speaker` audio
    df = df[df['speaker_minutes'] > seconds_per_speaker / 60.]
    speakers = df.drop_duplicates('speaker_id').sample(n_speakers, weights='speaker_minutes')['speaker_id']

    # For each of these speakers construct take a set of samples which has approximately the
    # right number of seconds.
    new_df = []
    for n, speaker in tqdm(enumerate(speakers)):
        df_ = df[df['speaker_id'] == speaker].sort_values('seconds', ascending=False)

        df_1 = df_[df_['seconds'].cumsum() < seconds_per_speaker]
        remaining_seconds = seconds_per_speaker - df_1['seconds'].sum()
        df_remaining = df_[df_['seconds'] < remaining_seconds].head(1)

        new_df.append(pd.concat([df_1, df_remaining]))

    new_df = pd.concat(new_df)

    new_dataset.df = new_df

    return new_dataset


#################
# Training Loop #
#################
train = LibriSpeechDataset(training_set, n_seconds, pad=pad)
valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False, pad=pad)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))


n_speakers = np.ceil(np.logspace(np.log(min_speakers), np.log(max_speakers), n_points, base=np.e)).astype(int)
for n in n_speakers:
    minutes_per_speaker = min_speakers*max_minutes*1./n
    seconds_per_speaker = minutes_per_speaker*60.
    print '*' * 35
    print '{} speakers, {:.2f} minutes per speaker'.format(n, min_speakers*max_minutes*1./n)
    print '*' * 35

    reduced_train = create_reduced_dataset(train, n, seconds_per_speaker)
    train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))

    for repeat in range(n_repeats):
        encoder = get_baseline_convolutional_encoder(model_n_filters, model_embedding_dimension, dropout=model_dropout)
        siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
        opt = Adam(clipnorm=1.)
        siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Train
        param_str = 'siamese__nspeakers_{}__secondsperspeaker_{}__filters_{}__embed_{}__drop_{}__r_{}'.format(
            n, int(seconds_per_speaker), model_n_filters, model_embedding_dimension, model_dropout, repeat)
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
                CSVLogger(PATH + '/logs/wide_vs_tall/{}.csv'.format(param_str)),
                ModelCheckpoint(
                    PATH + '/models/wide_vs_tall/{}.hdf5'.format(param_str),
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