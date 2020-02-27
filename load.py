from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import linecache

STEP = 256

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)


class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}
        self.labels = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = crop(x)
        means_expanded = np.outer(self.mean, np.ones(x[0].shape[1]))
        std_expaned = np.outer(self.std, np.ones(x[0].shape[1]))
        #x = (x - self.mean) / self.std
        x = (x - means_expanded) / std_expaned
        # x = x[:, :, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        # y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        # y = keras.utils.np_utils.to_categorical(
        #         y, num_classes=len(self.classes))

        y_vector = np.full((len(y), 9), 0)
        for i, label in enumerate(y):
            for j, ref in enumerate(self.labels):
                if ref in label:
                    y_vector[i, j] = 1

        return y_vector

def crop(x, val=0, dtype=np.float32):
    min_len = min(i.shape[1] for i in x)
    cropped = ()
    for e, i in enumerate(x):
        cropped += (i[:, :min_len],)
    return cropped

def pad(x, val=0, dtype=np.float32):
    max_len = max(i.shape[1] for i in x)
    padded = np.full((12, max_len), val, dtype=dtype).squeeze()
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x, axis=1).astype(np.float32),
           np.std(x, axis=1).astype(np.float32))

def load_dataset(directory):
    labels = []
    ecgs = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in tqdm.tqdm(files):
            if os.path.splitext(name)[1] == ".mat":
                patient = os.path.splitext(name)[0]
                ecg_file = os.path.join(root, name)
                label_file = os.path.join(root, patient+".hea")

                ecg = load_ecg(ecg_file)
                label = linecache.getline(label_file, 16)[5:-1]

                ecgs.append(ecg)
                labels.append(label)

    # with open(data_json, 'r') as fid:
    #     data = [json.loads(l) for l in fid]
    # for d in tqdm.tqdm(data_file):
    #     labels.append(d['labels'])
    #     ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()

    trunc_samp = STEP * int(ecg.shape[1] / STEP)
    return ecg[:, :trunc_samp]

if __name__ == "__main__":
    # data_json = "examples/cinc17/train.json"
    # train = load_dataset(data_json)
    data_directory = "Training_WFDB"
    train = load_dataset(data_directory)
    preproc = Preproc(*train)
    gen = data_generator(16, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break
