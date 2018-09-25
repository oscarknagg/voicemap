"""
This experiment tests the accuracy of a pre-trained siamese net on k-way, n-shot classification.

TODO:
- Test other baselines and pre-trained classifier bottleneck layers
"""
from keras.models import load_model
import pandas as pd
import os

from config import PATH
from voicemap.librispeech import LibriSpeechDataset
from voicemap.utils import BatchPreProcessor, preprocess_instances, evaluate_siamese_network


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
downsampling = 4
n_seconds = 3
validation_set = 'dev-clean'
siamese_model_path = PATH + '/models/n_seconds/siamese__nseconds_3.0__filters_128__embed_64__drop_0.0__r_0.hdf5'
classifier_model_path = PATH + '/models/baseline_classifier.hdf5'
k_way = range(2, 20, 1)
n_shot = [1, 5]
num_tasks = 1000
results_path = PATH + '/logs/k-way_n-shot_accuracy_{}.csv'.format(validation_set)


###################
# Create datasets #
###################
valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False)
batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))


#############
# Main Loop #
#############
siamese = load_model(siamese_model_path)
classifier = load_model(classifier_model_path)

with open(results_path, 'w') as f:
    print >>f, 'method,n_correct,n_tasks,n_shot,k_way'

results = []
for k in k_way:
    for n in n_shot:
        # print '*'*16
        # print '* {}-shot {}-way *'.format(n, k)
        # print '*'*16
        n_correct = evaluate_siamese_network(siamese, valid, batch_preprocessor, num_tasks, n, k)
        result = {'method': 'siamese', 'n_correct': n_correct, 'n_tasks': num_tasks, 'n': n, 'k': k}
        # print result
        results.append(result)

        with open(results_path, 'a') as f:
            print >>f, '{},{},{},{},{}'.format('siamese',n_correct,num_tasks,n,k)

        # # Classifier bottleneck
        # n_correct = evaluate_classification_network(classifier, valid, batch_preprocessor, num_tasks, n, k)
        # results.append({'method': 'bottleneck', 'n_correct': n_correct, 'n_tasks': num_tasks, 'n': n, 'k': k})

results = pd.DataFrame(results)
results.to_csv(results_path, index=False)
