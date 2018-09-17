"""
This experiment tests the accuracy of models trained on different length audio fragments.
"""
import os

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############
# Parameters #
##############
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


#################
# Training Loop #
#################
