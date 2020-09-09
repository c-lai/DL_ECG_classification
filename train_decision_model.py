import os
import numpy as np
from scipy.io import loadmat
from evaluate_12ECG_score import compute_beta_score
import load
from network_util import find_best_threshold, calculate_F_G, calculate_AUC
from scipy.io import savemat
import copy
from subset_selection import train_NN_classifier, train_tree_classifier
import json
import sys
import pickle


def permutation_importances_F1(rf, x, y, times):
    y_pred_bl = rf.predict(x)
    baseline_F1 = calculate_F_G(y_pred_bl, y, 1)[0]
    imp = []
    for t in range(times):
        for col in range(x.shape[1]):
            save = x[:, col].copy()
            x[:, col] = np.random.permutation(x[:, col])
            y_pred_permu = rf.predict(x)
            permu_F1 = calculate_F_G(y_pred_permu, y, 1)[0]
            x[:, col] = save
            imp.append(baseline_F1 - permu_F1)

    return np.array(imp).reshape((times, -1))


if __name__ == '__main__':
    # load data
    f = loadmat('.\\features\\features_train_final.mat')
    f_train = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_3_train'],
                              f['features_4_train'], f['features_5_train'], f['features_6_train'],
                              f['features_7_train'], f['features_8_train'], f['features_9_train'],
                              f['features_10_train'], f['features_11_train'], f['features_12_train']), axis=1)
    y_train = loadmat('.\\features\\y_train_final.mat')['y_train']

    f = loadmat('.\\features\\features_dev_final.mat')
    f_dev = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_3_dev'],
                            f['features_4_dev'], f['features_5_dev'], f['features_6_dev'],
                            f['features_7_dev'], f['features_8_dev'], f['features_9_dev'],
                            f['features_10_dev'], f['features_11_dev'], f['features_12_dev']), axis=1)
    y_dev = loadmat('.\\features\\y_dev_final.mat')['y_dev']

    f = loadmat('.\\features\\features_test_final.mat')
    f_test = np.concatenate((f['features_1_test'], f['features_2_test'], f['features_3_test'],
                             f['features_4_test'], f['features_5_test'], f['features_6_test'],
                             f['features_7_test'], f['features_8_test'], f['features_9_test'],
                             f['features_10_test'], f['features_11_test'], f['features_12_test']), axis=1)
    y_test = loadmat('.\\features\\y_test_final.mat')['y_test']

    subset = True
    model = 1
    ensemble = False
    experiment_name = 'decision_result_10_NN_v3_ori'

    lead_subsets = []
    train_pred_score_list = []
    train_pred_label_list = []
    val_pred_score_list = []
    val_pred_label_list = []
    test_pred_score_list = []
    test_pred_label_list = []
    if ensemble:
        for c in range(9):
            subset_result_file = 'forward_subset_selection_NN_10mean_F1_rhythm'+str(c)+'_v3.mat'
            file_path = os.path.join('.\\result\\subset_selection', subset_result_file)
            f = loadmat(file_path)
            leads = f['leads_selected'].flatten().tolist()
            p_value = f['p_value'].flatten()
            t_value = f['t_value'].flatten()

            lead_subset = []
            for i in range(12):
                if subset:
                    if p_value[i].item() <= 0.05 and t_value[i] >= 0:
                        lead_subset.append(leads[i])
                    else:
                        break
                else:
                    lead_subset.append(leads[i])
            lead_subsets.append(lead_subset)

            feature_index = []
            for l in lead_subset:
                feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
            feature_index = np.array(feature_index).flatten()

            model_folder = os.path.join('.\\save', experiment_name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_filepath = os.path.join(model_folder, 'model_rhythm' + str(c) + '.h5')

            input_dim = feature_index.shape[0]
            output_dim = 1
            if model == 1:
                NN_model, F1_train, G_train, AUC_train, \
                F1_val, G_val, AUC_val, \
                F1_test, G_test, AUC_test = \
                    train_NN_classifier(input_dim, output_dim,
                                        f_train[:, feature_index], y_train[:, c],
                                        f_dev[:, feature_index], y_dev[:, c],
                                        f_test[:, feature_index], y_test[:, c])
                NN_model.save(model_filepath)
                train_pred_score_c = np.asarray(NN_model.predict(f_train[:, feature_index]))
                val_pred_score_c = np.asarray(NN_model.predict(f_dev[:, feature_index]))
                test_pred_score_c = np.asarray(NN_model.predict(f_test[:, feature_index]))
            elif model == 2:
                tree_models, F1_train, G_train, AUC_train, \
                F1_val, G_val, AUC_val, \
                F1_test, G_test, AUC_test = \
                    train_tree_classifier(output_dim,
                                          f_train[:, feature_index], y_train[:, [c]],
                                          f_dev[:, feature_index], y_dev[:, [c]],
                                          f_test[:, feature_index], y_test[:, [c]])
                pickle.dump(tree_models[0], open(model_filepath, 'wb'))
                train_pred_score_c = tree_models[0].predict_proba(f_train[:, feature_index])[:, 1].reshape((-1, 1))
                val_pred_score_c = tree_models[0].predict_proba(f_dev[:, feature_index])[:, 1].reshape((-1, 1))
                test_pred_score_c = tree_models[0].predict_proba(f_test[:, feature_index])[:, 1].reshape((-1, 1))
            else:
                sys.exit('ERROR: No corresponding model (model=1: neural network; model=2: random forest)')

            train_pred_label_c = np.ceil(train_pred_score_c - 0.5)
            train_pred_score_list.append(train_pred_score_c)
            train_pred_label_list.append(train_pred_label_c)

            val_pred_label_c = np.ceil(val_pred_score_c - 0.5)
            val_pred_score_list.append(val_pred_score_c)
            val_pred_label_list.append(val_pred_label_c)

            test_pred_label_c = np.ceil(test_pred_score_c - 0.5)
            test_pred_score_list.append(test_pred_score_c)
            test_pred_label_list.append(test_pred_label_c)

        pred_score_train = np.concatenate(train_pred_score_list, axis=1)
        pred_label_train = np.concatenate(train_pred_label_list, axis=1)
        true_label_train = y_train
        pred_score_val = np.concatenate(val_pred_score_list, axis=1)
        pred_label_val = np.concatenate(val_pred_label_list, axis=1)
        true_label_val = y_dev
        pred_score_test = np.concatenate(test_pred_score_list, axis=1)
        pred_label_test = np.concatenate(test_pred_label_list, axis=1)
        true_label_test = y_test
    else:
        subset_result_file = 'forward_subset_selection_NN_10mean_F1_v3_ori.mat'
        file_path = os.path.join('.\\result\\subset_selection', subset_result_file)
        f = loadmat(file_path)
        leads = f['leads_selected'].flatten().tolist()
        p_value = f['p_value'].flatten()
        t_value = f['t_value'].flatten()

        lead_subset = []
        for i in range(12):
            if subset:
                if p_value[i].item() <= 0.05 and t_value[i] >= 0:
                    lead_subset.append(leads[i])
                else:
                    break
            else:
                lead_subset.append(leads[i])
        lead_subsets.append(lead_subset)

        feature_index = []
        for l in lead_subset:
            feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
        feature_index = np.array(feature_index).flatten()

        model_folder = os.path.join('.\\save', experiment_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_filepath = os.path.join(model_folder, 'NN_model.h5')

        input_dim = feature_index.shape[0]
        output_dim = 9

        NN_model, F1_train, G_train, AUC_train, \
        F1_val, G_val, AUC_val, \
        F1_test, G_test, AUC_test = \
            train_NN_classifier(input_dim, output_dim,
                                f_train[:, feature_index], y_train,
                                f_dev[:, feature_index], y_dev,
                                f_test[:, feature_index], y_test)
        NN_model.save(model_filepath)
        pred_score_train = np.asarray(NN_model.predict(f_train[:, feature_index]))
        pred_score_val = np.asarray(NN_model.predict(f_dev[:, feature_index]))
        pred_score_test = np.asarray(NN_model.predict(f_test[:, feature_index]))

        pred_label_train = np.ceil(pred_score_train - 0.5)
        pred_label_val = np.ceil(pred_score_val - 0.5)
        pred_label_test = np.ceil(pred_score_test - 0.5)

        true_label_train = y_train
        true_label_val = y_dev
        true_label_test = y_test

    accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
        compute_beta_score(true_label_train, pred_label_train, 1, 9)
    accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
        compute_beta_score(true_label_val, pred_label_val, 1, 9)
    accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
        compute_beta_score(true_label_test, pred_label_test, 1, 9)

    save_dir = os.path.join('.\\result', experiment_name+'.mat')
    savemat(save_dir,
            {'accuracy_train': accuracy_train,
             'F1_train': f_measure_train,
             'pred_score_train': pred_score_train,
             'pred_label_train': pred_label_train,
             'true_label_train': true_label_train,
             'accuracy_val': accuracy_val,
             'F1_val': f_measure_val,
             'pred_score_val': pred_score_val,
             'pred_label_val': pred_label_val,
             'true_label_val': true_label_val,
             'accuracy_test': accuracy_test,
             'F1_test': f_measure_test,
             'pred_score_test': pred_score_test,
             'pred_label_test': pred_label_test,
             'true_label_test': true_label_test})
    with open('.\\result\\lead_subsets_'+experiment_name+'.txt', 'w') as f:
        f.write(json.dumps(lead_subsets))



