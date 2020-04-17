from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import argparse
import json
import tensorflow as tf
import keras
import numpy as np

import random
import time

import network
import load
import util


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "epoch{epoch:03d}-val_loss{val_loss:.3f}-train_loss{loss:.3f}.hdf5")

def train(args, params):

    print("Loading training set...")
    train = load.load_dataset(params['train'], params['lead'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'], params['lead'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    # params.update({
    #     "input_shape": [None, 1],
    #     "num_categories": len(preproc.classes)
    # })

    # model = network.build_network(**params)
    model = network.build_network_1lead(**params)
    # model_path = ".\\save\\lead1_ResNet8_64_WMSE\\1586361669-991\\epoch017-val_loss0.392-train_loss0.179.hdf5"
    # model_path = ".\\save\\lead2_ResNet8_64_WMSE\\1586365719-237\\epoch015-val_loss0.251-train_loss0.202.hdf5"
    # model_path = ".\\save\\lead4_ResNet8_64_WMSE\\1586384117-946\\epoch021-val_loss0.466-train_loss0.194.hdf5"
    # model = keras.models.load_model(model_path,
    #                                 custom_objects={'weighted_mse': network.weighted_mse,
    #                                                 'weighted_cross_entropy': network.weighted_cross_entropy})

    stopping = keras.callbacks.EarlyStopping(patience=15)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=3,
        verbose=1,
        mode='min',
        min_lr=params["learning_rate"] * 0.01)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 4)

    metrics = network.Metrics_single_class(preproc.process(dev[0], dev[1]), batch_size=batch_size, save_dir = save_dir)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            class_weight=preproc.get_weight(),
            callbacks=[checkpointer, metrics, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
