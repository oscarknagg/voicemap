"""
This experiment determines the best verification distance threshold on the validation set and then determines uses this to
estimate the true verification accuracy on the test set.
"""
from keras.models import load_model
import pandas as pd
import os

from config import PATH
from voicemap.librispeech import LibriSpeechDataset
from voicemap.utils import BatchPreProcessor, preprocess_instances, n_shot_task_evaluation


# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    