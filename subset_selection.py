import os
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ReLU, BatchNormalization
from scipy.io import loadmat
from evaluate_12ECG_score import compute_beta_score
from keras.optimizers import Adam
import load
from train import get_filename_for_saving, make_save_dir
from network_util import Metrics_multi_class
from network_util import weighted_mse, weighted_binary_crossentropy
from network_util import find_best_threshold, calculate_F_G, calculate_AUC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.io import savemat
from scipy.stats import ttest_ind
import copy


beta = 1


def train_NN_classifier(input_dim, output_dim, x_train, y_train, x_val, y_val, x_test, y_test):
    lr = 0.001
    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=5,
        verbose=0,
        mode='min',
        min_lr=lr * 0.01)
    stopping = keras.callbacks.EarlyStopping(patience=20)

    NN_model = Sequential()
    NN_model.add(Dense(64, input_dim=input_dim))
    NN_model.add(BatchNormalization())
    NN_model.add(ReLU())
    NN_model.add(Dense(32))
    NN_model.add(BatchNormalization())
    NN_model.add(ReLU())
    NN_model.add(Dense(output_dim, activation='sigmoid'))
    NN_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr),
                     metrics=['accuracy'])

    weight0 = 0.5 / (1 - np.sum(y_train) / y_train.shape[0])
    weight1 = 0.5 / (np.sum(y_train) / y_train.shape[0])
    class_weight = {0: weight0, 1: weight1}
    NN_model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_data=(x_val, y_val),
                 # class_weight=class_weight,
                 callbacks=[reduce_lr, stopping], verbose=True)

    train_pred_score = np.asarray(NN_model.predict(x_train))
    train_targ = y_train.reshape((-1, output_dim))
    train_pred_label = np.ceil(train_pred_score - 0.5)
    if output_dim == 1:
        Fbeta_measure_train, Gbeta_measure_train, FG_mean_train = \
            calculate_F_G(train_pred_label, train_targ, beta)
        AUC_train = calculate_AUC(train_targ, train_pred_score)
    elif output_dim == 9:
        accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
            compute_beta_score(train_targ, train_pred_label, beta, 9)
        AUC_train = np.zeros((9, 1))
        for c in range(9):
            AUC_train[c, 0] = calculate_AUC(train_targ[:, c], train_pred_score[:, c])
        AUC_train = np.mean(AUC_train)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')

    val_pred_score = np.asarray(NN_model.predict(x_val))
    val_targ = y_val.reshape((-1, output_dim))
    val_pred_label = np.ceil(val_pred_score - 0.5)
    if output_dim == 1:
        Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = \
            calculate_F_G(val_pred_label, val_targ, beta)
        AUC_val = calculate_AUC(val_targ, val_pred_score)
    elif output_dim == 9:
        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
            compute_beta_score(val_targ, val_pred_label, beta, 9)
        AUC_val = np.zeros((9, 1))
        for c in range(9):
            AUC_val[c, 0] = calculate_AUC(val_targ[:, c], val_pred_score[:, c])
        AUC_val = np.mean(AUC_val)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')

    test_pred_score = np.asarray(NN_model.predict(x_test))
    test_targ = y_test.reshape((-1, output_dim))
    test_pred_label = np.ceil(test_pred_score - 0.5)
    if output_dim == 1:
        Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = \
            calculate_F_G(test_pred_label, test_targ, beta)
        AUC_test = calculate_AUC(test_targ, test_pred_score)
    elif output_dim == 9:
        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
            compute_beta_score(test_targ, test_pred_label, beta, 9)
        AUC_test = np.zeros((9, 1))
        for c in range(9):
            AUC_test[c, 0] = calculate_AUC(test_targ[:, c], test_pred_score[:, c])
        AUC_test = np.mean(AUC_test)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')

    return NN_model, Fbeta_measure_train, Gbeta_measure_train, AUC_train, \
           Fbeta_measure_val, Gbeta_measure_val, AUC_val, \
           Fbeta_measure_test, Gbeta_measure_test, AUC_test


def train_tree_classifier(output_dim, x_train, y_train, x_val, y_val, x_test, y_test):
    pred_label_train_list = []
    pred_label_val_list = []
    pred_label_test_list = []
    AUC_train = np.zeros((output_dim, 1))
    AUC_val = np.zeros((output_dim, 1))
    AUC_test = np.zeros((output_dim, 1))
    models = []
    for c in range(output_dim):
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5,
                                     bootstrap=True).fit(x_train, y_train[:, c])
        models.append(clf)

        pred_prob_c_train = clf.predict_proba(x_train)[:, 1]
        best_threshold = find_best_threshold(y_train[:, c], pred_prob_c_train, beta)

        # use the threshold to label subjects on training
        pred_prob_c_train = clf.predict_proba(x_train)[:, 1].reshape((-1, 1))
        pred_labels_c_train = np.ceil(pred_prob_c_train - best_threshold)
        pred_label_train_list.append(pred_labels_c_train)
        targ_labels_c_train = y_train[:, c].reshape((-1, 1))
        AUC_train[c, 0] = calculate_AUC(targ_labels_c_train, pred_prob_c_train)

        # use the threshold to label subjects on validation
        pred_prob_c_val = clf.predict_proba(x_val)[:, 1].reshape((-1, 1))
        pred_labels_c_val = np.ceil(pred_prob_c_val - best_threshold)
        pred_label_val_list.append(pred_labels_c_val)
        targ_labels_c_val = y_val[:, c].reshape((-1, 1))
        AUC_val[c, 0] = calculate_AUC(targ_labels_c_val, pred_prob_c_val)

        # use the threshold to label subjects on test
        pred_prob_c_test = clf.predict_proba(x_test)[:, 1].reshape((-1, 1))
        pred_labels_c_test = np.ceil(pred_prob_c_test - best_threshold)
        pred_label_test_list.append(pred_labels_c_test)
        targ_labels_c_test = y_test[:, c].reshape((-1, 1))
        AUC_test[c, 0] = calculate_AUC(targ_labels_c_test, pred_prob_c_test)

    train_pred_label = np.concatenate(pred_label_train_list, axis=1)
    train_targ = y_train.reshape((-1, output_dim))
    if output_dim == 1:
        Fbeta_measure_train, Gbeta_measure_train, FG_mean_train = \
            calculate_F_G(train_pred_label, train_targ, beta)
    elif output_dim == 9:
        accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
            compute_beta_score(train_targ, train_pred_label, beta, 9)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')
    AUC_train = AUC_train.mean()

    val_pred_label = np.concatenate(pred_label_val_list, axis=1)
    val_targ = y_val.reshape((-1, output_dim))
    if output_dim == 1:
        Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = \
            calculate_F_G(val_pred_label, val_targ, beta)
    elif output_dim == 9:
        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
            compute_beta_score(val_targ, val_pred_label, beta, 9)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')
    AUC_val = AUC_val.mean()

    test_pred_label = np.concatenate(pred_label_test_list, axis=1)
    test_targ = y_test.reshape((-1, output_dim))
    if output_dim == 1:
        Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = \
            calculate_F_G(test_pred_label, test_targ, beta)
    elif output_dim == 9:
        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
            compute_beta_score(test_targ, test_pred_label, beta, 9)
    else:
        sys.exit('ERROR: Only support subset selection based on 1 class or 9 classes')
    AUC_test = AUC_test.mean()

    return models, Fbeta_measure_train, Gbeta_measure_train, AUC_train, \
           Fbeta_measure_val, Gbeta_measure_val, AUC_val, \
           Fbeta_measure_test, Gbeta_measure_test, AUC_test


def forward_subset_selection(f_train, y_train, f_val, y_val, f_test, y_test,
                             repeat_times, rhythm_class, model, experiment_name):
    leads = list(range(12))
    leads_selection = []
    t_value = []
    p_value = []
    F1_all_train = []
    G_all_train = []
    AUC_all_train = []
    F1_all_val = []
    G_all_val = []
    AUC_all_val = []
    F1_all_test = []
    G_all_test = []
    AUC_all_test = []

    for i in range(12):
        print('Find top '+str(i+1)+' leads:')
        F1_train_lead = []
        G_train_lead = []
        AUC_train_lead = []
        F1_val_lead = []
        G_val_lead = []
        AUC_val_lead = []
        F1_test_lead = []
        G_test_lead = []
        AUC_test_lead = []
        for lead in leads:
            feature_index = [np.arange(32 * lead, 32 * (lead + 1), 1)]
            for l in leads_selection:
                feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
            feature_index = np.array(feature_index).flatten()

            F1_to_average_train = []
            G_to_average_train = []
            AUC_to_average_train = []
            F1_to_average_val = []
            G_to_average_val = []
            AUC_to_average_val = []
            F1_to_average_test = []
            G_to_average_test = []
            AUC_to_average_test = []

            for t in range(repeat_times):
                input_dim = 32*(i+1)
                output_dim = len(rhythm_class)
                if model == 1:
                    NN_model, F1_train, G_train, AUC_train, \
                    F1_val, G_val, AUC_val, \
                    F1_test, G_test, AUC_test = \
                        train_NN_classifier(input_dim, output_dim,
                                            f_train[:, feature_index], y_train[:, rhythm_class],
                                            f_val[:, feature_index], y_val[:, rhythm_class],
                                            f_test[:, feature_index], y_test[:, rhythm_class])
                elif model == 2:
                    tree_models, F1_train, G_train, AUC_train, \
                    F1_val, G_val, AUC_val, \
                    F1_test, G_test, AUC_test = \
                        train_tree_classifier(output_dim,
                                              f_train[:, feature_index], y_train[:, rhythm_class],
                                              f_val[:, feature_index], y_val[:, rhythm_class],
                                              f_test[:, feature_index], y_test[:, rhythm_class])
                else:
                    sys.exit('ERROR: No corresponding model (model=1: neural network; model=2: random forest)')

                F1_to_average_train.append(F1_train)
                G_to_average_train.append(G_train)
                AUC_to_average_train.append(AUC_train)
                F1_to_average_val.append(F1_val)
                G_to_average_val.append(G_val)
                AUC_to_average_val.append(AUC_val)
                F1_to_average_test.append(F1_test)
                G_to_average_test.append(G_test)
                AUC_to_average_test.append(AUC_test)

            F1_train_lead.append(F1_to_average_train)
            G_train_lead.append(G_to_average_train)
            AUC_train_lead.append(AUC_to_average_train)
            F1_val_lead.append(F1_to_average_val)
            G_val_lead.append(G_to_average_val)
            AUC_val_lead.append(AUC_to_average_val)
            F1_test_lead.append(F1_to_average_test)
            G_test_lead.append(G_to_average_test)
            AUC_test_lead.append(AUC_to_average_test)

        if i:
            t_value_val = np.zeros((12-i, 1))
            p_value_val = np.zeros((12-i, 1))
            for j, F1_val in enumerate(F1_val_lead):
                t, p = ttest_ind(np.array(F1_val), np.array(F1_all_val[-1]), equal_var=False)
                t_value_val[j] = t
                p_value_val[j] = p
            select_index = np.argmax(t_value_val)
            t_value.append(np.max(t_value_val))
            p_value.append(p_value_val[select_index])
        else:
            mean_F1 = np.concatenate(F1_val_lead).reshape([-1, repeat_times]).mean(axis=1)
            select_index = np.argmax(mean_F1)
            t_value.append(0)
            p_value.append(0)

        lead_selected = leads[select_index]
        leads_selection.append(lead_selected)
        leads.remove(lead_selected)
        F1_all_train.append(F1_train_lead[select_index])
        G_all_train.append(G_train_lead[select_index])
        AUC_all_train.append(AUC_train_lead[select_index])
        F1_all_val.append(F1_val_lead[select_index])
        G_all_val.append(G_val_lead[select_index])
        AUC_all_val.append(AUC_val_lead[select_index])
        F1_all_test.append(F1_test_lead[select_index])
        G_all_test.append(G_test_lead[select_index])
        AUC_all_test.append(AUC_test_lead[select_index])
        print(f"- lead:{lead_selected + 1:d} - t_value:{t_value[-1]:.3f}\n"
              f"- val_F1:{np.mean(F1_all_val[-1]):.3f} - val_G:{np.mean(G_all_val[-1]):.3f} "
              f"- val_AUC:{np.mean(AUC_all_val[-1]):.3f}\n"
              f"- test_F1:{np.mean(F1_all_test[-1]):.3f} - test_G:{np.mean(G_all_test[-1]):.3f} "
              f"- test_AUC:{np.mean(AUC_all_test[-1]):.3f}")

    save_dir = os.path.join('.\\result\\subset_selection', experiment_name)
    savemat(save_dir,
            {'leads_selected': np.array(leads_selection),
             't_value': np.array(t_value),
             'p_value': np.array(p_value),
             'F1_train': np.array(F1_all_train),
             'G_train': np.array(G_all_train),
             'AUC_train': np.array(AUC_all_train),
             'F1_val': np.array(F1_all_val),
             'G_val': np.array(G_all_val),
             'AUC_val': np.array(AUC_all_val),
             'F1_test': np.array(F1_all_test),
             'G_test': np.array(G_all_test),
             'AUC_test': np.array(AUC_all_test)})

    return


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

    repeat_time = 10
    rhythm_class = range(9)
    model = 1  # 1: NN; 2: random forest
    experiment_name = 'forward_subset_selection_NN_10mean_F1_v3.mat'

    forward_subset_selection(f_train, y_train, f_dev, y_dev, f_test, y_test,
                             repeat_time, rhythm_class, model, experiment_name)