from __future__ import print_function

import argparse
import numpy as np
import keras
import tensorflow as tf
import os

import load
import util

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs, labels = predict(args.data_path, args.model_path)
    np.save("val_prob", probs)
    np.save("val_labels", labels)
