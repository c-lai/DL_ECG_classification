import numpy as np
import keras
import os

import load
import util
from keras import Model
from network import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from network_util import calculate_F_G
from scipy.io import savemat
from evaluate_12ECG_score import compute_beta_score

# path for data
data_path_train = ".\\Training_WFDB\\train"
data_path_dev = ".\\Training_WFDB\\dev"
data_path_test = ".\\Training_WFDB\\test"

# path for lead models
leads = range(1, 13)
batch_size = 1
# leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE\\epoch022-val_loss0.393-train_loss0.316.hdf5",
#                     'lead2': ".\\save\\lead2_ResNet8_32_WCE\\epoch045-val_loss0.217-train_loss0.247.hdf5",
#                     'lead3': ".\\save\\lead3_ResNet8_32_WCE\\epoch024-val_loss0.291-train_loss0.329.hdf5",
#                     'lead4': ".\\save\\lead4_ResNet8_32_WCE\\epoch029-val_loss0.194-train_loss0.208.hdf5",
#                     'lead5': ".\\save\\lead5_ResNet8_32_WCE\\epoch030-val_loss0.354-train_loss0.319.hdf5",
#                     'lead6': ".\\save\\lead6_ResNet8_32_WCE\\epoch025-val_loss0.331-train_loss0.322.hdf5",
#                     'lead7': ".\\save\\lead7_ResNet8_32_WCE\\epoch030-val_loss0.315-train_loss0.298.hdf5",
#                     'lead8': ".\\save\\lead8_ResNet8_32_WCE\\epoch024-val_loss0.614-train_loss0.318.hdf5",
#                     'lead9': ".\\save\\lead9_ResNet8_32_WCE\\epoch023-val_loss0.278-train_loss0.287.hdf5",
#                     'lead10': ".\\save\\lead10_ResNet8_32_WCE\\epoch039-val_loss0.307-train_loss0.282.hdf5",
#                     'lead11': ".\\save\\lead11_ResNet8_32_WCE\\epoch027-val_loss0.485-train_loss0.320.hdf5",
#                     'lead12': ".\\save\\lead12_ResNet8_32_WCE\\epoch018-val_loss0.843-train_loss0.339.hdf5"}
# leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.557-train_loss0.156.hdf5",
#                     'lead2': ".\\save\\lead2_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.241-train_loss0.089.hdf5",
#                     'lead3': ".\\save\\lead3_ResNet8_NoDropout_32_WCE\\epoch022-val_loss0.279-train_loss0.124.hdf5",
#                     'lead4': ".\\save\\lead4_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.300-train_loss0.130.hdf5",
#                     'lead5': ".\\save\\lead5_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.389-train_loss0.206.hdf5",
#                     'lead6': ".\\save\\lead6_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.275-train_loss0.129.hdf5",
#                     'lead7': ".\\save\\lead7_ResNet8_NoDropout_32_WCE\\epoch008-val_loss0.738-train_loss0.226.hdf5",
#                     'lead8': ".\\save\\lead8_ResNet8_NoDropout_32_WCE\\epoch015-val_loss0.321-train_loss0.166.hdf5",
#                     'lead9': ".\\save\\lead9_ResNet8_NoDropout_32_WCE\\epoch052-val_loss0.307-train_loss0.183.hdf5",
#                     'lead10': ".\\save\\lead10_ResNet8_NoDropout_32_WCE\\epoch024-val_loss0.367-train_loss0.146.hdf5",
#                     'lead11': ".\\save\\lead11_ResNet8_NoDropout_32_WCE\\epoch020-val_loss0.494-train_loss0.175.hdf5",
#                     'lead12': ".\\save\\lead12_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.939-train_loss0.191.hdf5"}
leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE_v2\\epoch032-val_loss0.452-train_loss0.176.hdf5",
                    'lead2': ".\\save\\lead2_ResNet8_32_WCE_v2\\epoch049-val_loss0.016-train_loss0.153.hdf5",
                    'lead3': ".\\save\\lead3_ResNet8_32_WCE_v2\\epoch023-val_loss0.095-train_loss0.295.hdf5",
                    'lead4': ".\\save\\lead4_ResNet8_32_WCE_v2\\epoch076-val_loss0.033-train_loss0.187.hdf5",
                    'lead5': ".\\save\\lead5_ResNet8_32_WCE_v2\\epoch053-val_loss0.122-train_loss0.197.hdf5",
                    'lead6': ".\\save\\lead6_ResNet8_32_WCE_v2\\epoch024-val_loss0.068-train_loss0.213.hdf5",
                    'lead7': ".\\save\\lead7_ResNet8_32_WCE_v2\\epoch021-val_loss0.370-train_loss0.250.hdf5",
                    'lead8': ".\\save\\lead8_ResNet8_32_WCE_v2\\epoch021-val_loss0.363-train_loss0.277.hdf5",
                    'lead9': ".\\save\\lead9_ResNet8_32_WCE_v2\\epoch043-val_loss0.015-train_loss0.140.hdf5",
                    'lead10': ".\\save\\lead10_ResNet8_32_WCE_v2\\epoch100-val_loss0.050-train_loss0.199.hdf5",
                    'lead11': ".\\save\\lead11_ResNet8_32_WCE_v2\\epoch021-val_loss0.704-train_loss0.260.hdf5",
                    'lead12': ".\\save\\lead12_ResNet8_32_WCE_v2\\epoch022-val_loss0.452-train_loss0.246.hdf5"}

# load data
dataset_val = load.load_dataset(data_path_dev, False)
dataset_train = load.load_dataset(data_path_train, False)
dataset_test = load.load_dataset(data_path_test, False)

# load models and calculate metrics
F1_train = np.zeros((12, 9))
G_train = np.zeros((12, 9))
F1_val = np.zeros((12, 9))
G_val = np.zeros((12, 9))
F1_test = np.zeros((12, 9))
G_test = np.zeros((12, 9))
F1_lead_train = np.zeros((12, 1))
G_lead_train = np.zeros((12, 1))
F1_lead_val = np.zeros((12, 1))
G_lead_val = np.zeros((12, 1))
F1_lead_test = np.zeros((12, 1))
G_lead_test = np.zeros((12, 1))

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
    F1_lead_train[i, 0] = f_measure_train
    G_lead_train[i, 0] = Gbeta_measure_train
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_train_i[:, c], y_train_i[:, c], 1)
        F1_train[i, c] = F1_c
        G_train[i, c] = G_c

    val_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_val)
    y_val_i = np.empty((len(dataset_val[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_val[0]) / batch_size)):
        y_val_i[n, :] = next(val_gen_i)[1]
    y_pred_score_val_i = model_i.predict(val_gen_i, steps=int(len(dataset_val[0]) / batch_size), verbose=1)
    y_pred_label_val_i = np.ceil(y_pred_score_val_i - 0.5)
    accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
        compute_beta_score(y_val_i, y_pred_label_val_i, 1, 9)
    F1_lead_val[i, 0] = f_measure_val
    G_lead_val[i, 0] = Gbeta_measure_val
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_val_i[:, c], y_val_i[:, c], 1)
        F1_val[i, c] = F1_c
        G_val[i, c] = G_c

    test_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_test)
    y_test_i = np.empty((len(dataset_test[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_test[0]) / batch_size)):
        y_test_i[n, :] = next(test_gen_i)[1]
    y_pred_score_test_i = model_i.predict(test_gen_i, steps=int(len(dataset_test[0]) / batch_size), verbose=1)
    y_pred_label_test_i = np.ceil(y_pred_score_test_i - 0.5)
    accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
        compute_beta_score(y_test_i, y_pred_label_test_i, 1, 9)
    F1_lead_test[i, 0] = f_measure_test
    G_lead_test[i, 0] = Gbeta_measure_test
    for c in range(9):
        F1_c, G_c, FG_mean_c = calculate_F_G(y_pred_label_test_i[:, c], y_test_i[:, c], 1)
        F1_test[i, c] = F1_c
        G_test[i, c] = G_c

# save data
if not os.path.exists('.\\result'):
    os.makedirs('.\\result')

savemat('.\\result\\result_single_lead.mat',
        {'F1_train': F1_train,
         'G_train': G_train,
         'F1_lead_train': F1_lead_train,
         'G_lead_train': G_lead_train,
         'F1_val': F1_val,
         'G_val': G_val,
         'F1_lead_val': F1_lead_val,
         'G_lead_val': G_lead_val,
         'F1_test': F1_test,
         'G_test': G_test,
         'F1_lead_test': F1_lead_test,
         'G_lead_test': G_lead_test})