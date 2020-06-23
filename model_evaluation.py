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
model_path = ".\\save\\LSTM\\epoch016-val_loss1.836-train_loss1.016.hdf5"

# load data
dataset_val = load.load_dataset(data_path_dev, False)
dataset_train = load.load_dataset(data_path_train, False)
dataset_test = load.load_dataset(data_path_test, False)

# load model and calculate metrics
preproc = util.load(os.path.dirname(model_path))
model = keras.models.load_model(model_path)

x_train, y_train = preproc.process(*dataset_train)
y_pred_score_train = model.predict(x_train, verbose=1)
y_pred_label_train = np.ceil(y_pred_score_train - 0.5)
accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
    compute_beta_score(y_train, y_pred_label_train, 1, 9)
F1_train = f_measure_train
G_train = Gbeta_measure_train


x_val, y_val = preproc.process(*dataset_val)
y_pred_score_val = model.predict(x_val, verbose=1)
y_pred_label_val = np.ceil(y_pred_score_val - 0.5)
accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
    compute_beta_score(y_val, y_pred_label_val, 1, 9)
F1_val = f_measure_val
G_val = Gbeta_measure_val


x_test, y_test = preproc.process(*dataset_test)
y_pred_score_test = model.predict(x_test, verbose=1)
y_pred_label_test = np.ceil(y_pred_score_test - 0.5)
accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
    compute_beta_score(y_test, y_pred_label_test, 1, 9)
F1_test = f_measure_test
G_test = Gbeta_measure_test


# save data
savemat('.\\result\\result_LSTM.mat', {'F1_train': F1_train,
                            'G_train': G_train,
                            'F1_val': F1_val,
                            'G_val': G_val,
                            'F1_test': F1_test,
                            'G_test': G_test})
