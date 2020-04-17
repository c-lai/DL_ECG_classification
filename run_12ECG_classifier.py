#!/usr/bin/env python

import numpy as np
import os
import pickle
import keras
import tensorflow as tf
import util
import load
from network import weighted_mse, weighted_cross_entropy
from get_12ECG_features import get_12ECG_features
from scipy.io import loadmat

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

leads = [1, 2, 4]
lead1_model_path = "./save/lead1_final/epoch014-val_loss0.442-train_loss0.197.hdf5"
lead2_model_path = "./save/lead2_final/epoch020-val_loss0.044-train_loss0.109.hdf5"
lead4_model_path = "./save/lead4_final/epoch027-val_loss0.380-train_loss0.208.hdf5"

class1_final_model_path = "./save/decision_model_class1_final/epoch015-val_loss0.055-train_loss0.051.hdf5"
class2_final_model_path = "./save/decision_model_class2_final/epoch015-val_loss0.049-train_loss0.047.hdf5"
class3_final_model_path = "./save/decision_model_class3_final/epoch015-val_loss0.018-train_loss0.016.hdf5"
class4_final_model_path = "./save/decision_model_class4_final/epoch015-val_loss0.141-train_loss0.121.hdf5"
class5_final_model_path = "./save/decision_model_class5_final/epoch015-val_loss0.169-train_loss0.162.hdf5"
class6_final_model_path = "./save/decision_model_class6_final/epoch015-val_loss0.104-train_loss0.097.hdf5"
class7_final_model_path = "./save/decision_model_class7_final/epoch015-val_loss0.127-train_loss0.128.hdf5"
class8_final_model_path = "./save/decision_model_class8_final/epoch015-val_loss0.112-train_loss0.102.hdf5"
class9_final_model_path = "./save/decision_model_class9_final/epoch015-val_loss0.085-train_loss0.072.hdf5"

def ecg_standardize(ecg):
    ecg_mean = np.mean(ecg).astype(np.float32)
    ecg_std = np.std(ecg).astype(np.float32)
    s_ecg = (ecg - ecg_mean) / ecg_std
    return s_ecg

def run_12ECG_classifier(data, header_data, classes, model):
    preproc = util.load(os.path.dirname(lead1_model_path))
    dataset = (np.expand_dims(data, axis=0), header_data[15][5:-1])

    features_NN = []
    for i, lead in enumerate(leads):
        dataset_leadi = (ecg_standardize([dataset[0][:, lead-1, :]]), [dataset[1]])
        data_x, data_y = preproc.process(*dataset_leadi)
        feature_model = keras.Model(inputs=model[0][i].input, outputs=model[0][i].layers[-4].output)
        feature_NN_i = feature_model.predict(data_x, verbose=0)
        features_NN.append(feature_NN_i)

    features_benchmark = []
    feature_normalizer = loadmat("feature_normalizer.mat")
    for l in leads:
        feature_benchmark_l = (np.asarray(get_12ECG_features(data, header_data, l)) -
                               feature_normalizer['col_mean_'+str(l)])\
                              /feature_normalizer['col_std_'+str(l)]
        # feats_reshape_l = feature_benchmark_l.reshape(1, -1)
        features_benchmark.append(feature_benchmark_l[:, 2:])

    features = np.concatenate([np.concatenate(features_NN, axis=1),
                               feature_benchmark_l[:, :2],
                               np.concatenate(features_benchmark, axis=1)], axis=1)

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class.
    for c in range(num_classes):
        current_score_c = model[1][c]['model'].predict_proba(features)[0]
        current_score[c] = current_score_c
        current_label_c = np.ceil(current_score_c - model[1][c]['threshold'])
        current_label[c] = current_label_c

    return current_label, current_score

def load_threshold(directory):
    for file in os.listdir(directory):
        if os.path.splitext(file)[1] == ".txt":
            threshold = float(file[27:32])

    return threshold


def load_12ECG_model():
    # load the model from disk
    lead1_model = keras.models.load_model(lead1_model_path,
                                          custom_objects={'weighted_mse': weighted_mse,
                                                          'weighted_cross_entropy': weighted_cross_entropy})
    lead2_model = keras.models.load_model(lead2_model_path,
                                          custom_objects={'weighted_mse': weighted_mse,
                                                          'weighted_cross_entropy': weighted_cross_entropy})
    lead4_model = keras.models.load_model(lead4_model_path,
                                          custom_objects={'weighted_mse': weighted_mse,
                                                          'weighted_cross_entropy': weighted_cross_entropy})
    lead_models = (lead1_model, lead2_model, lead4_model)


    class1_final_model = {'model': keras.models.load_model(class1_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class1_final_model_path))}
    class2_final_model = {'model': keras.models.load_model(class2_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class2_final_model_path))}
    class3_final_model = {'model': keras.models.load_model(class3_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class3_final_model_path))}
    class4_final_model = {'model': keras.models.load_model(class4_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class4_final_model_path))}
    class5_final_model = {'model': keras.models.load_model(class5_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class5_final_model_path))}
    class6_final_model = {'model': keras.models.load_model(class6_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class6_final_model_path))}
    class7_final_model = {'model': keras.models.load_model(class7_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class7_final_model_path))}
    class8_final_model = {'model': keras.models.load_model(class8_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class8_final_model_path))}
    class9_final_model = {'model': keras.models.load_model(class9_final_model_path),
                          'threshold': load_threshold(os.path.dirname(class9_final_model_path))}

    final_models = (class1_final_model, class2_final_model, class3_final_model,
                    class4_final_model, class5_final_model, class6_final_model,
                    class7_final_model, class8_final_model, class9_final_model)

    loaded_models = (lead_models, final_models)

    return loaded_models
