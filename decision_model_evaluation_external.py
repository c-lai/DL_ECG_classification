import os
import numpy as np
import keras
from scipy.io import loadmat, savemat
from evaluate_12ECG_score import compute_beta_score
from network_util import find_best_threshold, calculate_F_G, calculate_AUC
from subset_selection import train_NN_classifier, train_tree_classifier
import json
import sys
import pickle


if __name__ == '__main__':
    # load data
    f = loadmat('.\\features\\features_external_test_E.mat')
    f_test = np.concatenate((f['features_1_test'], f['features_2_test'], f['features_3_test'],
                             f['features_4_test'], f['features_5_test'], f['features_6_test'],
                             f['features_7_test'], f['features_8_test'], f['features_9_test'],
                             f['features_10_test'], f['features_11_test'], f['features_12_test']), axis=1)
    y_test = loadmat('.\\features\\y_external_test_E.mat')['y_test']

    subset = True
    model = 1
    experiment_name = 'NN_rep10_F1'
    file_name = experiment_name
    if not subset:
        file_name = file_name+'_complete'
    save_folder = file_name + '_external_test_E'

    if subset:
        with open('.\\result\\subset_selection\\lead_subsets_'+experiment_name+'.txt') as f:
            lead_subset = json.load(f)[0]
    else:
        lead_subset = range(12)

    feature_index = []
    for l in lead_subset:
        feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
    feature_index = np.array(feature_index).flatten()

    model_folder = os.path.join('.\\save', 'decision_model_' + file_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    for root, dirs, files in os.walk(model_folder, topdown=False):
        for file in files:
            file_name_ext = os.path.splitext(file)
            if file_name_ext[1] == ".h5":
                model_name = file_name_ext[0]
                model_filepath = os.path.join(model_folder, model_name + '.h5')

                NN_model = keras.models.load_model(model_filepath)

                pred_score_test = np.asarray(NN_model.predict(f_test[:, feature_index]))
                pred_label_test = np.ceil(pred_score_test - 0.5)
                true_label_test = y_test

                # AF, AVB, BB, N, (PAC, PVC,) ST
                pred_score_test_adjust = np.zeros((pred_score_test.shape[0], 7))
                pred_label_test_adjust = np.zeros((pred_score_test.shape[0], 7))
                true_label_test_adjust = np.zeros((pred_score_test.shape[0], 7))

                pred_score_test_adjust[:, 0] = pred_score_test[:, 0]
                pred_score_test_adjust[:, 1] = pred_score_test[:, 1]
                pred_score_test_adjust[:, 2] = pred_score_test[:, 2] + pred_score_test[:, 6]
                pred_score_test_adjust[:, 3] = pred_score_test[:, 3]
                pred_score_test_adjust[:, 4] = pred_score_test[:, 4]
                pred_score_test_adjust[:, 5] = pred_score_test[:, 5]
                pred_score_test_adjust[:, 6] = pred_score_test[:, 7] + pred_score_test[:, 8]

                pred_label_test_adjust[:, 0] = pred_label_test[:, 0]
                pred_label_test_adjust[:, 1] = pred_label_test[:, 1]
                pred_label_test_adjust[:, 2] = pred_label_test[:, 2] + pred_label_test[:, 6]
                pred_label_test_adjust[:, 3] = pred_label_test[:, 3]
                pred_label_test_adjust[:, 4] = pred_label_test[:, 4]
                pred_label_test_adjust[:, 5] = pred_label_test[:, 5]
                pred_label_test_adjust[:, 6] = pred_label_test[:, 7] + pred_label_test[:, 8]

                true_label_test_adjust[:, 0] = true_label_test[:, 0] + true_label_test[:, 13]
                true_label_test_adjust[:, 1] = true_label_test[:, 1] + true_label_test[:, 12]
                true_label_test_adjust[:, 2] = true_label_test[:, 2] + true_label_test[:, 6] + true_label_test[:, 11]
                true_label_test_adjust[:, 3] = true_label_test[:, 3]
                true_label_test_adjust[:, 4] = true_label_test[:, 4]
                true_label_test_adjust[:, 5] = true_label_test[:, 5]
                true_label_test_adjust[:, 6] = true_label_test[:, 7] + true_label_test[:, 8] + \
                                               true_label_test[:, 9] + true_label_test[:, 10]

                pred_label_test_adjust = np.ceil(pred_label_test_adjust / 5)
                true_label_test_adjust = np.ceil(true_label_test_adjust / 5)

                PAC_idx = np.argwhere(true_label_test_adjust[:, 4] == 1)
                PVC_idx = np.argwhere(true_label_test_adjust[:, 5] == 1)
                PAC_PVC_idx = np.concatenate((PAC_idx, PVC_idx), axis=0)
                true_label_test_adjust = np.delete(true_label_test_adjust, PAC_PVC_idx, axis=0)
                true_label_test_adjust = np.delete(true_label_test_adjust, [4, 5], axis=1)
                pred_label_test_adjust = np.delete(pred_label_test_adjust, PAC_PVC_idx, axis=0)
                pred_label_test_adjust = np.delete(pred_label_test_adjust, [4, 5], axis=1)
                pred_score_test_adjust = np.delete(pred_score_test_adjust, PAC_PVC_idx, axis=0)
                pred_score_test_adjust = np.delete(pred_score_test_adjust, [4, 5], axis=1)

                accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
                    compute_beta_score(true_label_test_adjust, pred_label_test_adjust, 1, 5)

                save_dir = os.path.join('.\\result', 'external_test', save_folder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = os.path.join(save_dir, model_name + '.mat')
                savemat(save_file,
                        {'accuracy_test': accuracy_test,
                         'F1_test': f_measure_test,
                         'pred_score_test': pred_score_test_adjust,
                         'pred_label_test': pred_label_test_adjust,
                         'true_label_test': true_label_test_adjust})
