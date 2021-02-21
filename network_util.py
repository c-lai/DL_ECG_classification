import os
import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import confusion_matrix, roc_curve, auc
from evaluate_12ECG_score import compute_beta_score
import tensorflow as tf


def find_best_threshold(target, pred_score, beta):
    threshold = np.arange(0, 1, 0.01)

    tn = np.zeros(threshold.shape)
    fp = np.zeros(threshold.shape)
    fn = np.zeros(threshold.shape)
    tp = np.zeros(threshold.shape)
    for n, t in enumerate(threshold):
        tn[n], fp[n], fn[n], tp[n] = confusion_matrix(target, np.ceil(pred_score - t)).ravel()
    Fbeta_measure = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
    Gbeta_measure = tp / (tp + fp + beta * fn)
    FG_mean = (Fbeta_measure + Gbeta_measure) / 2
    best_threshold = threshold[np.argmax(Fbeta_measure)]

    return best_threshold


def calculate_F_G(pred_label, target, beta):
    tp = np.sum(pred_label * target)
    fp = np.sum(pred_label * (1 - target))
    fn = np.sum((1 - pred_label) * target)
    Fbeta_measure = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
    Gbeta_measure = tp / (tp + fp + beta * fn)
    FG_mean = (Fbeta_measure + Gbeta_measure) / 2

    return Fbeta_measure, Gbeta_measure, FG_mean


def calculate_AUC(target, pred_score):
    fpr, tpr, thresholds = roc_curve(target, pred_score)
    result_auc = auc(fpr, tpr)
    return result_auc


class Metrics_multi_class_from_generator(Callback):
    def __init__(self, train_gen, train_step, val_gen, val_step, test_gen, test_step, batch_size, save_dir):
        super().__init__()
        self.train_gen = train_gen
        self.train_step = train_step
        self.val_gen = val_gen
        self.val_step = val_step
        self.test_gen = test_gen
        self.test_step = test_step
        self.batch_size = batch_size
        self.save_dir = save_dir

        self.num_classes = 9
        self.beta = 1

    def on_train_begin(self, logs={}):
        self.val_accuracy = []
        self.val_f_measure = []
        self.val_Fbeta_measure = []
        self.val_Gbeta_measure = []

    def on_epoch_end(self, epoch, logs={}):
        # get true labels
        train_targ = np.empty((self.train_step * self.batch_size, 9), dtype=np.int64)
        for n in range(self.train_step):
            train_targ[n * self.batch_size:(n + 1) * self.batch_size, :] = next(self.train_gen)[1]
        val_targ = np.empty((self.val_step * self.batch_size, 9), dtype=np.int64)
        for n in range(self.val_step):
            val_targ[n * self.batch_size:(n + 1) * self.batch_size, :] = next(self.val_gen)[1]
        test_targ = np.empty((self.test_step * self.batch_size, 9), dtype=np.int64)
        for n in range(self.test_step):
            test_targ[n * self.batch_size:(n + 1) * self.batch_size, :] = next(self.test_gen)[1]
        # get predicted scores
        train_pred_score = np.asarray(self.model.predict(self.train_gen, steps=self.train_step))
        val_pred_score = np.asarray(self.model.predict(self.val_gen, steps=self.val_step))
        test_pred_score = np.asarray(self.model.predict(self.test_gen, steps=self.test_step))

        F_train = []
        F_val = []
        F_test = []
        best_threshold = []

        # get predicted labels and calculate metrics for each class
        train_pred_label = np.zeros((train_pred_score.shape[0], self.num_classes), dtype=int)
        val_pred_label = np.zeros((val_pred_score.shape[0], self.num_classes), dtype=int)
        test_pred_label = np.zeros((test_pred_score.shape[0], self.num_classes), dtype=int)
        for c in range(9):
            print(f'class {c:.0f}:', end=' ')

            # use 0.5 as threshold
            best_threshold_c = 0.5

            train_targ_c = train_targ[:, c].reshape((-1, 1))
            train_pred_score_c = train_pred_score[:, c].reshape((-1, 1))
            # # use the best threshold on training set
            # best_threshold_c = find_best_threshold(train_targ_c, train_pred_score_c, self.beta)

            best_threshold.append(best_threshold_c)

            # use the threshold to label subjects on training
            train_pred_label_c = np.ceil(train_pred_score_c - best_threshold_c)
            train_pred_label[:, c] = train_pred_label_c.reshape(-1)
            # use the threshold to label subjects on validation
            val_targ_c = val_targ[:, c].reshape((-1, 1))
            val_pred_score_c = val_pred_score[:, c].reshape((-1, 1))
            val_pred_label_c = np.ceil(val_pred_score_c - best_threshold_c)
            val_pred_label[:, c] = val_pred_label_c.reshape(-1)
            # use the threshold to label subjects on test
            test_targ_c = test_targ[:, c].reshape((-1, 1))
            test_pred_score_c = test_pred_score[:, c].reshape((-1, 1))
            test_pred_label_c = np.ceil(test_pred_score_c - best_threshold_c)
            test_pred_label[:, c] = test_pred_label_c.reshape(-1)

            # calculate metrics
            Fbeta_measure_train_c, Gbeta_measure_train_c, FG_mean_train_c = \
                calculate_F_G(train_pred_label_c, train_targ_c, self.beta)
            F_train.append(Fbeta_measure_train_c)
            Fbeta_measure_val_c, Gbeta_measure_val_c, FG_mean_val_c = \
                calculate_F_G(val_pred_label_c, val_targ_c, self.beta)
            F_val.append(Fbeta_measure_val_c)
            Fbeta_measure_test_c, Gbeta_measure_test_c, FG_mean_test_c = \
                calculate_F_G(test_pred_label_c, test_targ_c, self.beta)
            F_test.append(Fbeta_measure_test_c)

            print(f'{Fbeta_measure_val_c:.3f}(t:{best_threshold_c:.2f}),', end=' ')

        # calculate metrics for all classes
        accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train \
            = compute_beta_score(train_targ, train_pred_label, self.beta, self.num_classes)
        print("\n- train_f_measure:% f - train_Fbeta_measure:% f - train_Gbeta_measure:% f"
              % (f_measure_train, Fbeta_measure_train, Gbeta_measure_train))

        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val \
            = compute_beta_score(val_targ, val_pred_label, self.beta, self.num_classes)
        print("- val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f"
              % (f_measure_val, Fbeta_measure_val, Gbeta_measure_val))

        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test \
            = compute_beta_score(test_targ, test_pred_label, self.beta, self.num_classes)
        print("- test_f_measure:% f - test_Fbeta_measure:% f - test_Gbeta_measure:% f"
              % (f_measure_test, Fbeta_measure_test, Gbeta_measure_test))

        self.val_accuracy.append(accuracy_val)
        self.val_f_measure.append(f_measure_val)
        self.val_Fbeta_measure.append(Fbeta_measure_val)
        self.val_Gbeta_measure.append(Gbeta_measure_val)

        # save results
        with open(os.path.join(self.save_dir, f"log-epoch{epoch+1:03d}-F1{f_measure_val:.3f}.txt"), 'a', encoding='utf-8') as f:
            f.write(
                "train_accuracy:% f \ntrain_f_measure:% f \ntrain_Fbeta_measure:% f \ntrain_Gbeta_measure:% f \n"
                % (accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train))
            f.write(
                "val_accuracy:% f \nval_f_measure:% f \nval_Fbeta_measure:% f \nval_Gbeta_measure:% f \n"
                % (accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val))
            f.write(
                "test_accuracy:% f \ntest_f_measure:% f \ntest_Fbeta_measure:% f \ntest_Gbeta_measure:% f \n"
                % (accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test))
            for i in range(9):
                f.write(f'class {i+1:.0f}, threshold {best_threshold[i]:.3f} :\n'
                        f'F_train {F_train[i]:.3f},  F_val {F_val[i]:.3f}, F_test {F_test[i]:.3f},\n')

        return


class Metrics_multi_class(Callback):
    def __init__(self, train_data, val_data, test_data, save_dir):
        super().__init__()
        self.train_data = train_data
        self.validation_data = val_data
        self.test_data = test_data
        self.save_dir = save_dir

        self.num_classes = 9
        self.beta = 2

    def on_train_begin(self, logs={}):
        self.val_accuracy = []
        self.val_f_measure = []
        self.val_Fbeta_measure = []
        self.val_Gbeta_measure = []
        self.FG_mean = []

    def on_epoch_end(self, epoch, logs={}):
        train_pred_score = np.asarray(self.model.predict(self.train_data[0]))
        train_targ = self.train_data[1]
        val_pred_score = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]
        test_pred_score = np.asarray(self.model.predict(self.test_data[0]))
        test_targ = self.test_data[1]

        F_train = []
        F_val = []
        F_test = []
        best_threshold = []

        # get predicted labels and calculate metrics for each class
        train_pred_label = np.zeros((train_pred_score.shape[0], self.num_classes), dtype=int)
        val_pred_label = np.zeros((val_pred_score.shape[0], self.num_classes), dtype=int)
        test_pred_label = np.zeros((test_pred_score.shape[0], self.num_classes), dtype=int)
        for c in range(9):
            print(f'class {c:.0f}:', end=' ')

            # use 0.5 as threshold
            best_threshold_c = 0.5

            train_targ_c = train_targ[:, c].reshape((-1, 1))
            train_pred_score_c = train_pred_score[:, c].reshape((-1, 1))
            # # use the best threshold on training set
            # best_threshold_c = find_best_threshold(train_targ_c, train_pred_score_c, self.beta)

            best_threshold.append(best_threshold_c)

            # use the threshold to label subjects on training
            train_pred_label_c = np.ceil(train_pred_score_c - best_threshold_c)
            train_pred_label[:, c] = train_pred_label_c.reshape(-1)
            # use the threshold to label subjects on validation
            val_targ_c = val_targ[:, c].reshape((-1, 1))
            val_pred_score_c = val_pred_score[:, c].reshape((-1, 1))
            val_pred_label_c = np.ceil(val_pred_score_c - best_threshold_c)
            val_pred_label[:, c] = val_pred_label_c.reshape(-1)
            # use the threshold to label subjects on test
            test_targ_c = test_targ[:, c].reshape((-1, 1))
            test_pred_score_c = test_pred_score[:, c].reshape((-1, 1))
            test_pred_label_c = np.ceil(test_pred_score_c - best_threshold_c)
            test_pred_label[:, c] = test_pred_label_c.reshape(-1)

            # calculate metrics
            Fbeta_measure_train_c, Gbeta_measure_train_c, FG_mean_train_c = \
                calculate_F_G(train_pred_label_c, train_targ_c, self.beta)
            F_train.append(Fbeta_measure_train_c)
            Fbeta_measure_val_c, Gbeta_measure_val_c, FG_mean_val_c = \
                calculate_F_G(val_pred_label_c, val_targ_c, self.beta)
            F_val.append(Fbeta_measure_val_c)
            Fbeta_measure_test_c, Gbeta_measure_test_c, FG_mean_test_c = \
                calculate_F_G(test_pred_label_c, test_targ_c, self.beta)
            F_test.append(Fbeta_measure_test_c)

            print(f'{Fbeta_measure_val_c:.3f}(t:{best_threshold_c:.2f}),', end=' ')

        # calculate metrics for all classes
        accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train \
            = compute_beta_score(train_targ, train_pred_label, self.beta, self.num_classes)
        print("\n- train_f_measure:% f - train_Fbeta_measure:% f - train_Gbeta_measure:% f"
              % (f_measure_train, Fbeta_measure_train, Gbeta_measure_train))

        accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val \
            = compute_beta_score(val_targ, val_pred_label, self.beta, self.num_classes)
        print("- val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f"
              % (f_measure_val, Fbeta_measure_val, Gbeta_measure_val))

        accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test \
            = compute_beta_score(test_targ, test_pred_label, self.beta, self.num_classes)
        print("- test_f_measure:% f - test_Fbeta_measure:% f - test_Gbeta_measure:% f"
              % (f_measure_test, Fbeta_measure_test, Gbeta_measure_test))

        self.val_accuracy.append(accuracy_val)
        self.val_f_measure.append(f_measure_val)
        self.val_Fbeta_measure.append(Fbeta_measure_val)
        self.val_Gbeta_measure.append(Gbeta_measure_val)

        # save results
        with open(os.path.join(self.save_dir, f"log-epoch{epoch + 1:03d}-F1{f_measure_val:.3f}.txt"), 'a',
                  encoding='utf-8') as f:
            f.write(
                "train_accuracy:% f \ntrain_f_measure:% f \ntrain_Fbeta_measure:% f \ntrain_Gbeta_measure:% f \n"
                % (accuracy_train, f_measure_train, Fbeta_measure_train, Gbeta_measure_train))
            f.write(
                "val_accuracy:% f \nval_f_measure:% f \nval_Fbeta_measure:% f \nval_Gbeta_measure:% f \n"
                % (accuracy_val, f_measure_val, Fbeta_measure_val, Gbeta_measure_val))
            f.write(
                "test_accuracy:% f \ntest_f_measure:% f \ntest_Fbeta_measure:% f \ntest_Gbeta_measure:% f \n"
                % (accuracy_test, f_measure_test, Fbeta_measure_test, Gbeta_measure_test))
            for i in range(9):
                f.write(f'class {i + 1:.0f}, threshold {best_threshold[i]:.3f} :\n'
                        f'F_train {F_train[i]:.3f},  F_val {F_val[i]:.3f}, F_test {F_test[i]:.3f},\n')

        return


class Metrics_single_class(Callback):
    def __init__(self, train_data, val_data, test_data, save_dir):
        super().__init__()
        self.train_data = train_data
        self.validation_data = val_data
        self.test_data = test_data
        self.save_dir = save_dir

        self.beta = 1

    def on_train_begin(self, logs={}):
        self.val_accuracy = []
        self.val_f_measure = []
        self.val_Fbeta_measure = []
        self.val_Gbeta_measure = []
        self.FG_mean = []

    def on_epoch_end(self, epoch, logs={}):
        train_pred_score = np.asarray(self.model.predict(self.train_data[0]))
        train_targ = self.train_data[1].reshape((-1, 1))

        # # use 0.5 as threshold
        # best_threshold = 0.5

        # find the best threshold on training set
        best_threshold = find_best_threshold(train_targ, train_pred_score, self.beta)

        print(f"- beat threshold on train:{best_threshold:.3f}")

        # use the threshold to label subjects on training
        train_pred_score = np.asarray(self.model.predict(self.train_data[0]))
        train_targ = self.train_data[1].reshape((-1, 1))
        train_pred_label = np.ceil(train_pred_score - best_threshold)
        # use the threshold to label subjects on validation
        val_pred_score = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1].reshape((-1, 1))
        val_pred_label = np.ceil(val_pred_score - best_threshold)
        # use the threshold to label subjects on test
        test_pred_score = np.asarray(self.model.predict(self.test_data[0]))
        test_targ = self.test_data[1].reshape((-1, 1))
        test_pred_label = np.ceil(test_pred_score - best_threshold)

        # calculate measurements on validation
        Fbeta_measure_val, Gbeta_measure_val, FG_mean_val = calculate_F_G(val_pred_label, val_targ, self.beta)

        print("- val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
              % (Fbeta_measure_val, Gbeta_measure_val, FG_mean_val))

        with open(os.path.join(self.save_dir,
                               f"log-epoch{epoch + 1:03d}-beat_threshold{best_threshold:.3f}-FG_mean{FG_mean_val:.3f}.txt"),
                  'a', encoding='utf-8') as f:
            f.write("beat_threshold: %f \nval_Fbeta_measure:% f \nval_Gbeta_measure:% f \nGeometric Mean:% f"
                    % (best_threshold, Fbeta_measure_val, Gbeta_measure_val, FG_mean_val))

        return


# loss functions
def weighted_mse(yTrue, yPred):
    class_weight = K.constant([[0.043], [0.073], [0.264],
                               [0.057], [0.097], [0.084],
                               [0.031], [0.067], [0.284]])
    class_se = K.square(yPred-yTrue)
    w_mse = K.dot(class_se, class_weight)
    return K.sum(w_mse)


def weighted_cross_entropy(yTrue, yPred):
    class_weight = K.constant([[0.043], [0.073], [0.264],
                               [0.057], [0.097], [0.084],
                               [0.031], [0.067], [0.284]])
    class_CE = -(tf.math.multiply(yTrue, K.log(yPred))+tf.math.multiply(1-yTrue, K.log(1-yPred)))
    w_cross_entropy = K.dot(class_CE, class_weight)
    return K.sum(w_cross_entropy)


def weighted_binary_crossentropy(yTrue, yPred):
    class_weight = np.asarray([[0.6079384724186705, 2.8161343161343164], [0.5586515028432169, 4.762465373961219],
                               [0.5177684083722331,  14.569915254237289], [0.5770263467024669,  3.7456427015250546],
                               [0.5491934195815366,  5.58198051948052], [0.5566618099401004,  4.912142857142857],
                               [0.6849601593625498,  1.8516424340333872], [0.5723202396804261,  3.9568469505178365],
                               [0.5165239597416253,  15.629545454545454]])
    class_CE = -(yTrue * K.log(yPred) * class_weight[:, 1] + (1-yTrue) * K.log(1-yPred) * class_weight[:, 0])
    return K.mean(class_CE)


def weighted_binary_crossentropy_np(yTrue, yPred):
    class_weight = np.asarray([[0.6079384724186705, 2.8161343161343164], [0.5586515028432169, 4.762465373961219],
                               [0.5177684083722331, 14.569915254237289], [0.5770263467024669, 3.7456427015250546],
                               [0.5491934195815366, 5.58198051948052], [0.5566618099401004, 4.912142857142857],
                               [0.6849601593625498, 1.8516424340333872], [0.5723202396804261, 3.9568469505178365],
                               [0.5165239597416253, 15.629545454545454]])
    class_CE = -(yTrue * np.log(yPred) * class_weight[:, 1] + (1 - yTrue) * np.log(1 - yPred) * class_weight[:, 0])
    return np.mean(class_CE)