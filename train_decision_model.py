import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from scipy.io import loadmat
from evaluate_12ECG_score import compute_beta_score
from keras.optimizers import Adam
from train import get_filename_for_saving, make_save_dir
from network_util import Metrics_multi_class
from network_util import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from network_util import find_best_threshold, calculate_F_G
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def scheduler(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0001
    else:
        return 0.00001


def train_NN_model(f_train, y_train, f_dev, y_dev):
    print("Training neural network model")
    # set model
    model = Sequential()
    model.add(Dense(32, input_dim=32 * 12, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='sigmoid'))

    lr = 0.0001
    # model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.compile(loss=weighted_binary_crossentropy, optimizer=Adam(lr=lr), metrics=['categorical_accuracy'])
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    save_dir = "save"
    model_save_dir = make_save_dir(save_dir, "decision_model_all")

    stopping = keras.callbacks.EarlyStopping(patience=20)

    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=3,
        verbose=1,
        mode='min',
        min_lr=lr * 0.01)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(model_save_dir),
        save_best_only=False)

    metrics = Metrics_multi_class(train_data=(f_train, y_train), val_data=(f_dev, y_dev), save_dir=model_save_dir)

    model.fit(f_train, y_train, epochs=30, batch_size=10, validation_data=(f_dev, y_dev),
              callbacks=[checkpointer, metrics, reduce_lr, stopping])

    return model


def train_tree_model(f_train, y_train, f_dev, y_dev):
    print("Training tree models")

    beta = 2
    models = []
    pred_label_list = []
    for i in range(9):
        print("processing class %d/9..." % (i + 1))
        # clf = RandomForestClassifier(n_estimators=150, max_depth=3,
        #                              bootstrap=True, random_state=0).fit(f_train, y_train[:, i])
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=0).fit(f_train, y_train[:, i])
        models.append(clf)

        # find the best threshold on training set
        pred_prob_i_train = clf.predict_proba(f_train)[:, 1]
        FG_mean_train, best_threshold = find_best_threshold(y_train[:, i], pred_prob_i_train, beta)

        pred_prob_i_val = clf.predict_proba(f_dev)[:, 1].reshape((-1, 1))
        pred_labels_i = np.ceil(pred_prob_i_val - best_threshold)

        # use the threshold to label subjects on validation
        # pred_labels_i = clf.predict(f_dev).reshape((-1, 1))
        pred_label_list.append(pred_labels_i)
        targ_labels_i = y_dev[:, i].reshape((-1, 1))

        # calculate measurements on validation
        Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = calculate_F_G(pred_labels_i, targ_labels_i, beta)

        print("- val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
              % (Fbeta_measure_val, Gbeta_measure_val, FG_mean_val))

    targ_label = y_dev
    pred_label = np.concatenate(pred_label_list, axis=1)
    accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(targ_label, pred_label, beta, 9)
    FG_mean = np.mean([Fbeta_measure, Gbeta_measure])
    print(
        "All: - val_accuracy:% f - val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
        % (accuracy, f_measure, Fbeta_measure, Gbeta_measure, FG_mean))

    return models


# load data
f = loadmat('features_train_final.mat')
f_train = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_3_train'],
                          f['features_4_train'], f['features_5_train'], f['features_6_train'],
                          f['features_7_train'], f['features_8_train'], f['features_9_train'],
                          f['features_10_train'], f['features_11_train'], f['features_12_train']), axis=1)
# f_train = np.concatenate((f['features_2_train'], f['features_4_train'], f['features_7_train'],
#                           f['features_9_train'], f['features_10_train']), axis=1)
y_train = loadmat('y_train_final.mat')['y_train']

f = loadmat('features_dev_final.mat')
f_dev = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_3_dev'],
                        f['features_4_dev'], f['features_5_dev'], f['features_6_dev'],
                        f['features_7_dev'], f['features_8_dev'], f['features_9_dev'],
                        f['features_10_dev'], f['features_11_dev'], f['features_12_dev']), axis=1)
# f_dev = np.concatenate((f['features_2_dev'], f['features_4_dev'], f['features_7_dev'],
#                         f['features_9_dev'], f['features_10_dev']), axis=1)
y_dev = loadmat('y_dev_final.mat')['y_dev']

NN_model = train_NN_model(f_train, y_train, f_dev, y_dev)
tree_models = train_tree_model(f_train, y_train, f_dev, y_dev)

