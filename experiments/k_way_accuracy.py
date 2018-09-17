"""
This experiment tests the accuracy of a pre-trained siamese net on k-way, n-shot classification.

TODO:
- Test other baselines and pre-trained classifier bottleneck layers
"""
from keras.models import load_model
import pandas as pd

from config import PATH
from voicemap.librispeech import LibriSpeechDataset
from voicemap.utils import BatchPreProcessor, preprocess_instances, evaluate_siamese_network_nshot

##############
# Parameters #
##############
downsampling = 4
n_seconds = 3
validation_set = 'dev-clean'
siamese_model_path = PATH + '/models/baseline_convnet.hdf5'
classifier_model_path = PATH + '/models/baseline_classifier.hdf5'
k_way = range(2, 20, 1)
n_shot = [1, 5]
num_tasks = 1000


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

results = []
for n in n_shot:
    for k in k_way:
        n_correct = evaluate_siamese_network_nshot(siamese, valid, batch_preprocessor, num_tasks, n, k)
        results.append({'method': 'siamese', 'n_correct': n_correct, 'n_tasks': num_tasks, 'n': n, 'k': k})

        # # Classifier bottleneck
        # n_correct = evaluate_classification_network(classifier, valid, batch_preprocessor, num_tasks, n, k)
        # results.append({'method': 'bottleneck', 'n_correct': n_correct, 'n_tasks': num_tasks, 'n': n, 'k': k})

        # # DFT frequencies
        # results.append({'method': 'dft', 'n_correct': n_correct, 'n_tasks': num_tasks, 'n': n, 'k': k})

results = pd.DataFrame(results)
results.to_csv(PATH + '/logs/k-way_n-shot_accuracy_{}.csv'.format(validation_set), index=False)
