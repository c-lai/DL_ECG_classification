# Find useful data on the external test set, move them to data/external_test

import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import linecache
from shutil import copyfile

STEP = 512

def process_label(label):
    # SNOMED-CT codes
    # (https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv)
    labels = ("164889003", "270492004", "164909002", "426783006", "59118001", "284470004",
              "164884008", "429622005", "164931005",
              "164930006", "55930002", "6374002", "233917008", "195080001")
    y_vector = np.full((len(labels)), 0)
    for j, ref in enumerate(labels):
        if ref in label:
            y_vector[j] = 1

    return y_vector

def check_dataset(directory):
    patients = []
    test_directory = "data/external_test"
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in tqdm.tqdm(files):
            if os.path.splitext(name)[1] == ".mat":
                patient = os.path.splitext(name)[0]
                ecg_file = os.path.join(root, name)
                label_file = os.path.join(root, patient+".hea")

                ecg = load_ecg(ecg_file)
                ecg_mean = np.mean(ecg, axis=1).astype(np.float32)
                ecg_std = (np.std(ecg, axis=1)+0.001).astype(np.float32)
                means_expanded = np.outer(ecg_mean, np.ones(ecg.shape[1]))
                std_expanded = np.outer(ecg_std, np.ones(ecg.shape[1]))
                ecg = (ecg - means_expanded) / std_expanded

                label = linecache.getline(label_file, 16)[5:-1]
                y_vector = process_label(label)

                if np.sum(y_vector) > 0:
                    patients.append(patient)
                    copyfile(os.path.join(directory, patient + ".hea"),
                             os.path.join(test_directory, patient + ".hea"))
                    copyfile(os.path.join(directory, patient + ".mat"),
                             os.path.join(test_directory, patient + ".mat"))

    with open(os.path.join(directory, 'check_no_sinus.txt'), 'w') as f:
        f.write('\n'.join(patients))

    return

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()

    # trunc_samp = STEP * min([int(ecg.shape[1] / STEP), 8])
    trunc_samp = STEP * int(ecg.shape[1] / STEP)
    return ecg[:, :trunc_samp]

if __name__ == "__main__":
    data_directory = "data/PhysioNetChallenge2020_Training_StPetersburg"
    check_dataset(data_directory)
