import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ReLU
from scipy.io import loadmat
from evaluate_12ECG_score import compute_beta_score
from keras.optimizers import Adam
from train import get_filename_for_saving, make_save_dir
from network_util import Metrics_multi_class
from network_util import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy, weighted_binary_crossentropy_np
from network_util import find_best_threshold, calculate_F_G, calculate_AUC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.io import savemat
import copy


def scheduler(epoch):
    if epoch < 15:
        return 0.0001
    else:
        return 0.00001


def train_NN_model(f_train, y_train, f_dev, y_dev, f_test, y_test):
    beta = 2

    print("Training neural network model")
    # set model
    model = Sequential()
    model.add(Dense(64, input_dim=32 * 5))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
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

    y_test_pred_score = model.predict(f_test)
    y_test_pred_class = np.ceil(y_test_pred_score - 0.5)
    accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
        compute_beta_score(y_test, y_test_pred_class, beta, 9)
    FG_mean_test = np.mean([Fbeta_measure_test, Gbeta_measure_test])
    # print(
    #     "test_accuracy:% f - test_f_measure:% f - test_Fbeta_measure:% f - test_Gbeta_measure:% f - Geometric Mean:% f"
    #     % (accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test, FG_mean_test))
    # Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = calculate_F_G(y_test_pred_class, y_test, beta)
    AUC_test = np.zeros((9, 1))
    for c in range(9):
        AUC_test[c, 0] = calculate_AUC(y_test[:, c], y_test_pred_score[:, c])
    print("- test_accuracy:% f - test_F1:% f - test_Fbeta_measure:% f - test_Gbeta_measure:% f - Geometric Mean:% f - test_AUC:% f"
          % (accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test, FG_mean_test, np.mean(AUC_test)))


    return model


def train_tree_model(f_train, y_train, f_dev, y_dev, f_test, y_test):
    print("Training tree models")

    beta = 2
    models = []
    pred_label_val_list = []
    pred_label_test_list = []
    important_leads = np.zeros((1, 9))
    for i in range(9):
        print("processing class %d/9..." % (i + 1))
        clf = RandomForestClassifier(n_estimators=500, max_depth=4,
                                     bootstrap=True, random_state=0).fit(f_train, y_train[:, i])
        # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
        #                                  max_depth=3, random_state=0).fit(f_train, y_train[:, i])
        models.append(clf)

        # importance = clf.feature_importances_
        # lead_importance = np.sum(importance.reshape((12, -1)), axis=1)
        # important_lead = np.argmax(lead_importance)+1
        # important_leads[0, i] = important_lead
        # print(f"important lead:{important_lead:d} - importance:{np.max(lead_importance):.03f}")

        # find the best threshold on training set
        pred_prob_i_train = clf.predict_proba(f_train)[:, 1]
        FG_mean_train, best_threshold = find_best_threshold(y_train[:, i], pred_prob_i_train, beta)

        # use the threshold to label subjects on validation
        pred_prob_i_val = clf.predict_proba(f_dev)[:, 1].reshape((-1, 1))
        pred_labels_i_val = np.ceil(pred_prob_i_val - best_threshold)
        # pred_labels_i_val = clf.predict(f_dev).reshape((-1, 1))
        pred_label_val_list.append(pred_labels_i_val)
        targ_labels_i_val = y_dev[:, i].reshape((-1, 1))

        # use the threshold to label subjects on test
        pred_prob_i_test = clf.predict_proba(f_test)[:, 1].reshape((-1, 1))
        pred_labels_i_test = np.ceil(pred_prob_i_test - best_threshold)
        # pred_labels_i_test = clf.predict(f_test).reshape((-1, 1))
        pred_label_test_list.append(pred_labels_i_test)
        targ_labels_i_test = y_test[:, i].reshape((-1, 1))

        # calculate measurements on validation
        Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = calculate_F_G(pred_labels_i_val, targ_labels_i_val, beta)
        # calculate measurements on test
        Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = calculate_F_G(pred_labels_i_test, targ_labels_i_test, beta)

        print("- val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
              % (Fbeta_measure_val, Gbeta_measure_val, FG_mean_val))
        print("- test_Fbeta_measure:% f - test_Gbeta_measure:% f - Geometric Mean:% f"
              % (Fbeta_measure_test, Gbeta_measure_test, FG_mean_test))

    # calculate overall measurements
    pred_label_val = np.concatenate(pred_label_val_list, axis=1)
    accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
        compute_beta_score(y_dev, pred_label_val, beta, 9)
    FG_mean_val = np.mean([Fbeta_measure_val, Gbeta_measure_val])
    print(
        "All: - val_accuracy:% f - val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
        % (accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val, FG_mean_val))

    pred_label_test = np.concatenate(pred_label_test_list, axis=1)
    accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
        compute_beta_score(y_test, pred_label_test, beta, 9)
    FG_mean_test = np.mean([Fbeta_measure_test, Gbeta_measure_test])
    print(
        "All: - test_accuracy:% f - test_f_measure:% f - test_Fbeta_measure:% f - test_Gbeta_measure:% f - Geometric Mean:% f"
        % (accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test, FG_mean_test))

    return models


def backward_subset_selection(f_train, y_train, f_dev, y_dev, f_test, y_test):
    leads = list(range(12))
    leads_selection = []
    WCE_all = []
    FG_mean_all_val = []
    F1_all_val = []
    F2_all_val = []
    G_all_val = []
    AUC_all_val = []
    FG_mean_all_test = []
    F1_all_test = []
    F2_all_test = []
    G_all_test = []
    AUC_all_test = []

    stopping = keras.callbacks.EarlyStopping(patience=10)
    for i in range(11):
        print('Remove '+str(i+1)+' leads:')
        loss = []
        FG_mean_val_lead = []
        F1_val_lead = []
        F2_val_lead = []
        G_val_lead = []
        AUC_val_lead = []
        FG_mean_test_lead = []
        F1_test_lead = []
        F2_test_lead = []
        G_test_lead = []
        AUC_test_lead = []
        for lead in leads:
            leads_remain = copy.deepcopy(leads)
            leads_remain.remove(lead)
            feature_index = []
            for l in leads_remain:
                feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
            feature_index = np.array(feature_index).flatten()

            loss_to_average = []
            FG_mean_to_average = []
            F1_to_average = []
            F2_to_average = []
            G_to_average = []
            AUC_to_average = []
            FG_mean_to_average_test = []
            F1_to_average_test = []
            F2_to_average_test = []
            G_to_average_test = []
            AUC_to_average_test = []
            for j in range(5):
                redo = True
                lr = 0.0001
                while(redo):
                    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
                    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                        factor=0.1,
                        patience=3,
                        verbose=0,
                        mode='min',
                        min_lr=lr * 0.01)

                    model = Sequential()
                    model.add(Dense(64, input_dim=32 * (11-i)))
                    model.add(ReLU())
                    model.add(Dense(32))
                    model.add(ReLU())
                    model.add(Dense(9, activation='sigmoid'))
                    model.compile(loss=weighted_binary_crossentropy, optimizer=Adam(lr=lr), metrics=['categorical_accuracy'])

                    model.fit(f_train[:, feature_index], y_train, epochs=30, batch_size=10, validation_data=(f_dev[:, feature_index], y_dev),
                              callbacks=[reduce_lr, stopping], verbose=False)

                    results = model.evaluate(x=f_dev[:, feature_index], y=y_dev, verbose=False)
                    if np.isnan(results[0]):
                        lr = lr*0.5
                    else:
                        redo = False
                        val_pred_score = np.asarray(model.predict(f_dev[:, feature_index]))
                        val_targ = y_dev
                        val_pred_label = np.ceil(val_pred_score - 0.5)
                        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
                            compute_beta_score(val_targ, val_pred_label, 1, 9)
                        FG_mean_val = np.mean([Fbeta_measure_val, Gbeta_measure_val])
                        AUC_val = np.zeros((9, 1))
                        for c in range(9):
                            AUC_val[c, 0] = calculate_AUC(val_targ[:, c], val_pred_score[:, c])
                        AUC_val = np.mean(AUC_val)
                        # Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = calculate_F_G(val_pred_label, val_targ, 2)

                        test_pred_score = np.asarray(model.predict(f_test[:, feature_index]))
                        test_targ = y_test
                        test_pred_label = np.ceil(test_pred_score - 0.5)
                        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
                            compute_beta_score(test_targ, test_pred_label, 1, 9)
                        FG_mean_test = np.mean([Fbeta_measure_test, Gbeta_measure_test])
                        AUC_test = np.zeros((9, 1))
                        for c in range(9):
                            AUC_test[c, 0] = calculate_AUC(test_targ[:, c], test_pred_score[:, c])
                        AUC_test = np.mean(AUC_test)
                        # Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = calculate_F_G(test_pred_label, test_targ, 2)

                loss_to_average.append(results[0])
                FG_mean_to_average.append(FG_mean_val)
                F1_to_average.append(f_measure_val)
                F2_to_average.append(Fbeta_measure_val)
                G_to_average.append(Gbeta_measure_val)
                AUC_to_average.append(AUC_val)
                FG_mean_to_average_test.append(FG_mean_test)
                F1_to_average_test.append(f_measure_test)
                F2_to_average_test.append(Fbeta_measure_test)
                G_to_average_test.append(Gbeta_measure_test)
                AUC_to_average_test.append(AUC_test)

            loss.append(np.mean(loss_to_average))
            FG_mean_val_lead.append(np.mean(FG_mean_to_average))
            F1_val_lead.append(np.mean(F1_to_average))
            F2_val_lead.append(np.mean(F2_to_average))
            G_val_lead.append(np.mean(G_to_average))
            AUC_val_lead.append(np.mean(AUC_to_average))
            FG_mean_test_lead.append(np.mean(FG_mean_to_average_test))
            F1_test_lead.append(np.mean(F1_to_average_test))
            F2_test_lead.append(np.mean(F2_to_average_test))
            G_test_lead.append(np.mean(G_to_average_test))
            AUC_test_lead.append(np.mean(AUC_to_average_test))

        select_index = np.argmax(F1_val_lead)
        lead_selected = leads[select_index]
        leads_selection.append(lead_selected)
        leads.remove(lead_selected)
        WCE_all.append(loss[select_index])
        FG_mean_all_val.append(FG_mean_val_lead[select_index])
        F1_all_val.append(F1_val_lead[select_index])
        F2_all_val.append(F2_val_lead[select_index])
        G_all_val.append(G_val_lead[select_index])
        AUC_all_val.append(AUC_val_lead[select_index])
        FG_mean_all_test.append(FG_mean_test_lead[select_index])
        F1_all_test.append(F1_test_lead[select_index])
        F2_all_test.append(F2_test_lead[select_index])
        G_all_test.append(G_test_lead[select_index])
        AUC_all_test.append(AUC_test_lead[select_index])
        print(f"lead:{lead_selected + 1:d} - val_loss:{WCE_all[-1]:.3f}\n"
              f"-val_F1:{F1_all_val[-1]:.3f}-val_F2:{F2_all_val[-1]:.3f}-val_G:{G_all_val[-1]:.3f}"
              f"-val_FG_mean:{FG_mean_all_val[-1]:.3f}-val_AUC:{AUC_all_val[-1]:.3f}\n"
              f"test_F1:{F1_all_test[-1]:.3f}-test_F2:{F2_all_test[-1]:.3f}-test_G:{G_all_test[-1]:.3f}"
              f"-test_FG_mean:{FG_mean_test_lead[select_index]:.3f}-test_AUC:{AUC_all_test[-1]:.3f}")

    savemat('backward_subset_selection_5mean_dropout_relu_F1.mat',
            {'leads_selected': np.array(leads_selection),
             'WCE': np.array(WCE_all),
             'FG_mean_all_val': np.array(FG_mean_all_val),
             'F1_val': np.array(F1_all_val),
             # 'F2_val': np.array(F2_all_val),
             'G_val': np.array(G_all_val),
             'AUC_val': np.array(AUC_all_val),
             'FG_mean_all_test': np.array(FG_mean_all_test),
             'F1_test': np.array(F1_all_test),
             # 'F2_test': np.array(F2_all_test),
             'G_test': np.array(G_all_test),
             'AUC_test': np.array(AUC_all_test)})
    plt.subplot(311)
    plt.plot(FG_mean_all_val, label='FG_mean_val')
    plt.plot(F2_all_val, label='F_val')
    plt.plot(G_all_val, label='G_val')
    plt.subplot(312)
    plt.plot(FG_mean_all_test, label='FG_mean_test')
    plt.plot(F2_all_test, label='F_test')
    plt.plot(G_all_test, label='G_test')
    plt.subplot(313)
    plt.plot(WCE_all, label='loss_val')
    plt.show()


def forward_subset_selection(f_train, y_train, f_dev, y_dev, f_test, y_test):
    leads = list(range(12))
    leads_selection = []
    WCE_all = []
    FG_mean_all_train = []
    F1_all_train = []
    F2_all_train = []
    G_all_train = []
    AUC_all_train = []
    FG_mean_all_val = []
    F1_all_val = []
    F2_all_val = []
    G_all_val = []
    AUC_all_val = []
    FG_mean_all_test = []
    F1_all_test = []
    F2_all_test = []
    G_all_test = []
    AUC_all_test = []

    stopping = keras.callbacks.EarlyStopping(patience=10)
    for i in range(12):
        print('Find top '+str(i+1)+' leads:')
        FG_mean_train_lead = []
        F1_train_lead = []
        F2_train_lead = []
        G_train_lead = []
        AUC_train_lead = []
        loss = []
        FG_mean_val_lead = []
        F1_val_lead = []
        F2_val_lead = []
        G_val_lead = []
        AUC_val_lead = []
        FG_mean_test_lead = []
        F1_test_lead = []
        F2_test_lead = []
        G_test_lead = []
        AUC_test_lead = []
        for lead in leads:
            feature_index = [np.arange(32 * lead, 32 * (lead + 1), 1)]
            for l in leads_selection:
                feature_index.append(np.arange(32 * l, 32 * (l + 1), 1))
            feature_index = np.array(feature_index).flatten()

            FG_mean_to_average_train = []
            F1_to_average_train = []
            F2_to_average_train = []
            G_to_average_train = []
            AUC_to_average_train = []
            loss_to_average = []
            FG_mean_to_average_val = []
            F1_to_average_val = []
            F2_to_average_val = []
            G_to_average_val = []
            AUC_to_average_val = []
            FG_mean_to_average_test = []
            F1_to_average_test = []
            F2_to_average_test = []
            G_to_average_test = []
            AUC_to_average_test = []
            for j in range(5):
                redo = True
                lr = 0.0001
                while(redo):
                    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
                    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                        factor=0.1,
                        patience=3,
                        verbose=0,
                        mode='min',
                        min_lr=lr * 0.01)

                    model = Sequential()
                    model.add(Dense(64, input_dim=32 * (i + 1)))
                    model.add(ReLU())
                    model.add(Dense(32))
                    model.add(ReLU())
                    model.add(Dense(9, activation='sigmoid'))
                    model.compile(loss=weighted_binary_crossentropy, optimizer=Adam(lr=lr), metrics=['categorical_accuracy'])

                    model.fit(f_train[:, feature_index], y_train, epochs=30, batch_size=10, validation_data=(f_dev[:, feature_index], y_dev),
                              callbacks=[reduce_lr, stopping], verbose=False)

                    results = model.evaluate(x=f_dev[:, feature_index], y=y_dev, verbose=False)
                    if np.isnan(results[0]):
                        lr = lr * 0.1
                    else:
                        redo = False
                        train_pred_score = np.asarray(model.predict(f_dev[:, feature_index]))
                        train_targ = y_dev
                        train_pred_label = np.ceil(train_pred_score - 0.5)
                        accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train = \
                            compute_beta_score(train_targ, train_pred_label, 1, 9)
                        FG_mean_train = np.mean([Fbeta_measure_train, Gbeta_measure_train])
                        AUC_train = np.zeros((9, 1))
                        for c in range(9):
                            AUC_train[c, 0] = calculate_AUC(train_targ[:, c], train_pred_score[:, c])
                        AUC_train = np.mean(AUC_train)
                        # Fbeta_measure_train, Gbeta_measure_train, FG_mean_train = calculate_F_G(train_pred_label, train_targ, 2)

                        val_pred_score = np.asarray(model.predict(f_train[:, feature_index]))
                        val_targ = y_train
                        val_pred_label = np.ceil(val_pred_score - 0.5)
                        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val = \
                            compute_beta_score(val_targ, val_pred_label, 1, 9)
                        FG_mean_val = np.mean([Fbeta_measure_val, Gbeta_measure_val])
                        AUC_val = np.zeros((9, 1))
                        for c in range(9):
                            AUC_val[c, 0] = calculate_AUC(val_targ[:, c], val_pred_score[:, c])
                        AUC_val = np.mean(AUC_val)
                        # Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = calculate_F_G(val_pred_label, val_targ, 2)

                        test_pred_score = np.asarray(model.predict(f_test[:, feature_index]))
                        test_targ = y_test
                        test_pred_label = np.ceil(test_pred_score - 0.5)
                        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test = \
                            compute_beta_score(test_targ, test_pred_label, 1, 9)
                        FG_mean_test = np.mean([Fbeta_measure_test, Gbeta_measure_test])
                        AUC_test = np.zeros((9, 1))
                        for c in range(9):
                            AUC_test[c, 0] = calculate_AUC(test_targ[:, c], test_pred_score[:, c])
                        AUC_test = np.mean(AUC_test)
                        # Fbeta_measure_test, Gbeta_measure_test, FG_mean_test = calculate_F_G(test_pred_label, test_targ, 2)

                loss_to_average.append(results[0])
                FG_mean_to_average_train.append(FG_mean_train)
                F1_to_average_train.append(f_measure_train)
                F2_to_average_train.append(Fbeta_measure_train)
                G_to_average_train.append(Gbeta_measure_train)
                AUC_to_average_train.append(AUC_train)
                FG_mean_to_average_val.append(FG_mean_val)
                F1_to_average_val.append(f_measure_val)
                F2_to_average_val.append(Fbeta_measure_val)
                G_to_average_val.append(Gbeta_measure_val)
                AUC_to_average_val.append(AUC_val)
                FG_mean_to_average_test.append(FG_mean_test)
                F1_to_average_test.append(f_measure_test)
                F2_to_average_test.append(Fbeta_measure_test)
                G_to_average_test.append(Gbeta_measure_test)
                AUC_to_average_test.append(AUC_test)

            loss.append(np.mean(loss_to_average))
            FG_mean_train_lead.append(np.mean(FG_mean_to_average_train))
            F1_train_lead.append(np.mean(F1_to_average_train))
            F2_train_lead.append(np.mean(F2_to_average_train))
            G_train_lead.append(np.mean(G_to_average_train))
            AUC_train_lead.append(np.mean(AUC_to_average_train))
            FG_mean_val_lead.append(np.mean(FG_mean_to_average_val))
            F1_val_lead.append(np.mean(F1_to_average_val))
            F2_val_lead.append(np.mean(F2_to_average_val))
            G_val_lead.append(np.mean(G_to_average_val))
            AUC_val_lead.append(np.mean(AUC_to_average_val))
            FG_mean_test_lead.append(np.mean(FG_mean_to_average_test))
            F1_test_lead.append(np.mean(F1_to_average_test))
            F2_test_lead.append(np.mean(F2_to_average_test))
            G_test_lead.append(np.mean(G_to_average_test))
            AUC_test_lead.append(np.mean(AUC_to_average_test))

        select_index = np.argmax(F1_val_lead)
        lead_selected = leads[select_index]
        leads_selection.append(lead_selected)
        leads.remove(lead_selected)
        WCE_all.append(loss[select_index])
        FG_mean_all_train.append(FG_mean_train_lead[select_index])
        F1_all_train.append(F1_train_lead[select_index])
        F2_all_train.append(F2_train_lead[select_index])
        G_all_train.append(G_train_lead[select_index])
        AUC_all_train.append(AUC_train_lead[select_index])
        FG_mean_all_val.append(FG_mean_val_lead[select_index])
        F1_all_val.append(F1_val_lead[select_index])
        F2_all_val.append(F2_val_lead[select_index])
        G_all_val.append(G_val_lead[select_index])
        AUC_all_val.append(AUC_val_lead[select_index])
        FG_mean_all_test.append(FG_mean_test_lead[select_index])
        F1_all_test.append(F1_test_lead[select_index])
        F2_all_test.append(F2_test_lead[select_index])
        G_all_test.append(G_test_lead[select_index])
        AUC_all_test.append(AUC_test_lead[select_index])
        print(f"lead:{lead_selected + 1:d} - val_loss:{WCE_all[-1]:.3f}\n"
              f"-val_F1:{F1_all_val[-1]:.3f}-val_F2:{F2_all_val[-1]:.3f}-val_G:{G_all_val[-1]:.3f}"
              f"-val_FG_mean:{FG_mean_all_val[-1]:.3f}-val_AUC:{AUC_all_val[-1]:.3f}\n"
              f"test_F1:{F1_all_test[-1]:.3f}-test_F2:{F2_all_test[-1]:.3f}-test_G:{G_all_test[-1]:.3f}"
              f"-test_FG_mean:{FG_mean_test_lead[select_index]:.3f}-test_AUC:{AUC_all_test[-1]:.3f}")

    savemat('forward_subset_selection_5mean_dropout_relu_F1_final.mat',
            {'leads_selected': np.array(leads_selection),
             'WCE': np.array(WCE_all),
             'FG_mean_all_train': np.array(FG_mean_all_train),
             'F1_train': np.array(F1_all_train),
             # 'F2_train': np.array(F2_all_train),
             'G_train': np.array(G_all_train),
             'AUC_train': np.array(AUC_all_train),
             'FG_mean_all_val': np.array(FG_mean_all_val),
             'F1_val': np.array(F1_all_val),
             # 'F2_val': np.array(F2_all_val),
             'G_val': np.array(G_all_val),
             'AUC_val': np.array(AUC_all_val),
             'FG_mean_all_test': np.array(FG_mean_all_test),
             'F1_test': np.array(F1_all_test),
             # 'F2_test': np.array(F2_all_test),
             'G_test': np.array(G_all_test),
             'AUC_test': np.array(AUC_all_test)})
    plt.subplot(311)
    plt.plot(FG_mean_all_val, label='FG_mean_val')
    plt.plot(F2_all_val, label='F_val')
    plt.plot(G_all_val, label='G_val')
    plt.subplot(312)
    plt.plot(FG_mean_all_test, label='FG_mean_test')
    plt.plot(F2_all_test, label='F_test')
    plt.plot(G_all_test, label='G_test')
    plt.subplot(313)
    plt.plot(WCE_all, label='loss_val')
    plt.show()


# load data
f = loadmat('features_train.mat')
f_train = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_3_train'],
                          f['features_4_train'], f['features_5_train'], f['features_6_train'],
                          f['features_7_train'], f['features_8_train'], f['features_9_train'],
                          f['features_10_train'], f['features_11_train'], f['features_12_train']), axis=1)
# f_train = np.concatenate((f['features_3_train'], f['features_4_train'], f['features_7_train'],
#                           f['features_9_train'], f['features_10_train']), axis=1)
y_train = loadmat('y_train.mat')['y_train']

f = loadmat('features_dev.mat')
f_dev = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_3_dev'],
                        f['features_4_dev'], f['features_5_dev'], f['features_6_dev'],
                        f['features_7_dev'], f['features_8_dev'], f['features_9_dev'],
                        f['features_10_dev'], f['features_11_dev'], f['features_12_dev']), axis=1)
# f_dev = np.concatenate((f['features_3_dev'], f['features_4_dev'], f['features_7_dev'],
#                         f['features_9_dev'], f['features_10_dev']), axis=1)
y_dev = loadmat('y_dev.mat')['y_dev']

f = loadmat('features_test.mat')
f_test = np.concatenate((f['features_1_test'], f['features_2_test'], f['features_3_test'],
                         f['features_4_test'], f['features_5_test'], f['features_6_test'],
                         f['features_7_test'], f['features_8_test'], f['features_9_test'],
                         f['features_10_test'], f['features_11_test'], f['features_12_test']), axis=1)
# f_test = np.concatenate((f['features_3_test'], f['features_4_test'], f['features_7_test'],
#                          f['features_9_test'], f['features_10_test']), axis=1)
y_test = loadmat('y_test.mat')['y_test']

# NN_model = train_NN_model(f_train, y_train, f_dev, y_dev, f_test, y_test)
# tree_models = train_tree_model(f_train, y_train, f_dev, y_dev, f_test, y_test)
forward_subset_selection(f_train, y_train, f_dev, y_dev, f_test, y_test)
# backward_subset_selection(f_train, y_train, f_dev, y_dev, f_test, y_test)

