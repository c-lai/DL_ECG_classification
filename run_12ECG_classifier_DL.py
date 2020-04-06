#!/usr/bin/env python

import numpy as np
import os
import pickle
import keras
import tensorflow as tf
import util
from network import weighted_mse, weighted_cross_entropy

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_path = ".\\save\\lead6_ResNet16_WMSE\\1586184908-757\\epoch030-val_loss0.139-train_loss0.273.hdf5"
lead = 6

def run_12ECG_classifier(data, header_data, classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class.
    preproc = util.load(os.path.dirname(model_path))
    dataset = (np.expand_dims(data, axis=0), header_data[15][5:-1])
    data_x, data_y = preproc.process(*dataset)

    score = model.predict(data_x)
    label = np.argmax(score)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score


def load_preproc(dirname):
    preproc = util.load(dirname)
    return preproc


def load_12ECG_model():
    # load the model from disk
    loaded_model = keras.models.load_model(model_path,
                                           custom_objects={'weighted_mse': weighted_mse,
                                                           'weighted_cross_entropy': weighted_cross_entropy})

    return loaded_model
