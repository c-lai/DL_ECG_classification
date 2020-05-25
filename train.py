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
import network_util

# set random seed
# for reproducible results, optimizer can't be Adam (can use RMSprop instead)
seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# configure GPU
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

    random.seed(seed_value)

    # model = network.build_network_ResNet(**params)
    model = network.build_network_LSTM(**params)

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

    metrics = network_util.Metrics_multi_class(preproc.process(train[0], train[1]), preproc.process(dev[0], dev[1]),
                                               save_dir=save_dir)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            # class_weight=preproc.get_weight(), #for single class models
            callbacks=[checkpointer, metrics, reduce_lr, stopping],
            shuffle=True)
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, metrics, reduce_lr, stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
