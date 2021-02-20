# Evaluate single-lead models
# Instructions: change leads_model_path and save path

import numpy as np
import keras
import os

import load
import util
from keras import Model
from network_util import calculate_F_G
from network_util import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from scipy.io import savemat
from evaluate_12ECG_score import compute_beta_score

# path for data
data_path_train = ".\\data_old\\train"
data_path_dev = ".\\data_old\\dev"
data_path_test = ".\\data_old\\test"

# path for lead models
leads = range(1, 13)
batch_size = 1
leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE\\epoch022-val_loss0.393-train_loss0.316.hdf5",
                    'lead2': ".\\save\\lead2_ResNet8_32_WCE\\epoch045-val_loss0.217-train_loss0.247.hdf5",
                    'lead3': ".\\save\\lead3_ResNet8_32_WCE\\epoch024-val_loss0.291-train_loss0.329.hdf5",
                    'lead4': ".\\save\\lead4_ResNet8_32_WCE\\epoch029-val_loss0.194-train_loss0.208.hdf5",
                    'lead5': ".\\save\\lead5_ResNet8_32_WCE\\epoch030-val_loss0.354-train_loss0.319.hdf5",
                    'lead6': ".\\save\\lead6_ResNet8_32_WCE\\epoch025-val_loss0.331-train_loss0.322.hdf5",
                    'lead7': ".\\save\\lead7_ResNet8_32_WCE\\epoch030-val_loss0.315-train_loss0.298.hdf5",
                    'lead8': ".\\save\\lead8_ResNet8_32_WCE\\epoch024-val_loss0.614-train_loss0.318.hdf5",
                    'lead9': ".\\save\\lead9_ResNet8_32_WCE\\epoch023-val_loss0.278-train_loss0.287.hdf5",
                    'lead10': ".\\save\\lead10_ResNet8_32_WCE\\epoch039-val_loss0.307-train_loss0.282.hdf5",
                    'lead11': ".\\save\\lead11_ResNet8_32_WCE\\epoch027-val_loss0.485-train_loss0.320.hdf5",
                    'lead12': ".\\save\\lead12_ResNet8_32_WCE\\epoch018-val_loss0.843-train_loss0.339.hdf5"}

# load data
dataset_train = load.load_dataset(data_path_train, False)
dataset_val = load.load_dataset(data_path_dev, False)
dataset_test = load.load_dataset(data_path_test, False)

# load models and calculate metrics
F1_train_leadxrhythm = np.zeros((12, 9))
G_train_leadxrhythm = np.zeros((12, 9))
F1_val_leadxrhythm = np.zeros((12, 9))
G_val_leadxrhythm = np.zeros((12, 9))
F1_test_leadxrhythm = np.zeros((12, 9))
G_test_leadxrhythm = np.zeros((12, 9))
F1_train_lead = np.zeros((12, 1))
G_train_lead = np.zeros((12, 1))
F1_val_lead = np.zeros((12, 1))
G_val_lead = np.zeros((12, 1))
F1_test_lead = np.zeros((12, 1))
G_test_lead = np.zeros((12, 1))

for i, lead in enumerate(leads):
    preproc_i = util.load(os.path.dirname(leads_model_path['lead'+str(lead)]))
    model_i = keras.models.load_model(leads_model_path['lead'+str(lead)],
                                      custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

    train_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_train)
    y_train_i = np.empty((len(dataset_train[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_train[0]) / batch_size)):
        y_train_i[n, :] = next(train_gen_i)[1]
    y_pred_score_train_i = model_i.predict(train_gen_i, steps=int(len(dataset_train[0]) / batch_size), verbose=1)
    y_pred_label_train_i = np.ceil(y_pred_score_train_i - 0.5)
    accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
        compute_beta_score(y_train_i, y_pred_label_train_i, 1, 9)
    F1_train_lead[i, 0] = f_measure_train
    G_train_lead[i, 0] = Gbeta_measure_train
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_train_i[:, c], y_train_i[:, c], 1)
        F1_train_leadxrhythm[i, c] = F1_c
        G_train_leadxrhythm[i, c] = G_c

    val_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_val)
    y_val_i = np.empty((len(dataset_val[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_val[0]) / batch_size)):
        y_val_i[n, :] = next(val_gen_i)[1]
    y_pred_score_val_i = model_i.predict(val_gen_i, steps=int(len(dataset_val[0]) / batch_size), verbose=1)
    y_pred_label_val_i = np.ceil(y_pred_score_val_i - 0.5)
    accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
        compute_beta_score(y_val_i, y_pred_label_val_i, 1, 9)
    F1_val_lead[i, 0] = f_measure_val
    G_val_lead[i, 0] = Gbeta_measure_val
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_val_i[:, c], y_val_i[:, c], 1)
        F1_val_leadxrhythm[i, c] = F1_c
        G_val_leadxrhythm[i, c] = G_c

    test_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_test)
    y_test_i = np.empty((len(dataset_test[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_test[0]) / batch_size)):
        y_test_i[n, :] = next(test_gen_i)[1]
    y_pred_score_test_i = model_i.predict(test_gen_i, steps=int(len(dataset_test[0]) / batch_size), verbose=1)
    y_pred_label_test_i = np.ceil(y_pred_score_test_i - 0.5)
    accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
        compute_beta_score(y_test_i, y_pred_label_test_i, 1, 9)
    F1_test_lead[i, 0] = f_measure_test
    G_test_lead[i, 0] = Gbeta_measure_test
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_test_i[:, c], y_test_i[:, c], 1)
        F1_test_leadxrhythm[i, c] = F1_c
        G_test_leadxrhythm[i, c] = G_c

# save data
if not os.path.exists('.\\result'):
    os.makedirs('.\\result')

savemat('.\\result\\result_single_lead_model.mat',
        {'F1_train_leadxrhythm': F1_train_leadxrhythm,
         'G_train_leadxrhythm': G_train_leadxrhythm,
         'F1_train_lead': F1_train_lead,
         'G_train_lead': G_train_lead,
         'F1_val_leadxrhythm': F1_val_leadxrhythm,
         'G_val_leadxrhythm': G_val_leadxrhythm,
         'F1_val_lead': F1_val_lead,
         'G_val_lead': G_val_lead,
         'F1_test_leadxrhythm': F1_test_leadxrhythm,
         'G_test_leadxrhythm': G_test_leadxrhythm,
         'F1_test_lead': F1_test_lead,
         'G_test_lead': G_test_lead})
