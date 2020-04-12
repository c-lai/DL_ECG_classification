# first neural network with keras tutorial
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from scipy.io import loadmat
from evaluate_12ECG_score import compute_beta_score
from keras.optimizers import Adam
from network import weighted_mse, weighted_cross_entropy
from train import get_filename_for_saving, make_save_dir
from sklearn.metrics import confusion_matrix

# f = loadmat('features_train.mat')
# f_train = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_4_train']), axis=1)
# # f_train = f['features_train']
# y_train = loadmat('y_train.mat')['y_train']
#
# f = loadmat('features_dev.mat')
# f_dev = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_4_dev']), axis=1)
# # f_dev = f['features_dev']
# y_dev = loadmat('y_dev.mat')['y_dev']


f_1 = loadmat('features_all_final.mat')
# f_all_1 = np.concatenate((f['features_1_all'], f['features_2_all'], f['features_4_all']), axis=1)
f_2 = loadmat('features_bm_all_standard.mat')
# f_all_2 = np.concatenate((f['features_1_all'], f['features_2_all'], f['features_4_all']), axis=1)
f_train = np.concatenate((f_1['features_1_all'], f_1['features_2_all'], f_1['features_4_all'],
                          f_2['features_bm_1_all_standard'], f_2['features_bm_2_all_standard'][:, 2:],
                          f_2['features_bm_4_all_standard'][:, 2:]), axis=1)
# f_train = f['features_train']
y_train = loadmat('y_all_final.mat')['y_all']

f_1 = loadmat('features_all_final.mat')
# f_dev_1 = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_4_dev']), axis=1)
f_2 = loadmat('features_bm_all_standard.mat')
# f_dev_2 = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_4_dev']), axis=1)
f_dev = np.concatenate((f_1['features_1_all'], f_1['features_2_all'], f_1['features_4_all'],
                        f_2['features_bm_1_all_standard'], f_2['features_bm_2_all_standard'][:, 2:],
                        f_2['features_bm_4_all_standard'][:, 2:]), axis=1)
# f_all = f['features_all']
y_dev = loadmat('y_all_final.mat')['y_all']


save_dir = "save"

model = Sequential()
model.add(Dense(64, input_dim=230, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(9, activation='sigmoid'))


class Metrics(Callback):
    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = save_dir

    def on_train_begin(self, logs={}):
        self.num_classes = 9
        self.beta = 2
        self.val_accuracy = []
        self.val_f_measure = []
        self.val_Fbeta_measure = []
        self.val_Gbeta_measure = []
        self.FG_mean = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred_score = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]

        val_pred_label = np.zeros((val_pred_score.shape[0], self.num_classes), dtype=int)
        labels = np.argmax(val_pred_score, axis=1)
        for i, label in enumerate(labels):
            val_pred_label[i, label] = 1

        accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(val_targ, val_pred_label, self.beta,
                                                                               self.num_classes)
        FG_mean = np.mean([Fbeta_measure, Gbeta_measure])
        self.val_accuracy.append(accuracy)
        self.val_f_measure.append(f_measure)
        self.val_Fbeta_measure.append(Fbeta_measure)
        self.val_Gbeta_measure.append(Gbeta_measure)
        self.FG_mean.append(FG_mean)
        print(" - val_accuracy:% f - val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
              % (accuracy, f_measure, Fbeta_measure, Gbeta_measure, FG_mean))

        # threshold = np.arange(0, 1, 0.001)
        # tn = np.zeros(threshold.shape)
        # fp = np.zeros(threshold.shape)
        # fn = np.zeros(threshold.shape)
        # tp = np.zeros(threshold.shape)
        # for n, t in enumerate(threshold):
        #     tn[n], fp[n], fn[n], tp[n] = confusion_matrix(val_targ, np.ceil(val_pred_score - t)).ravel()
        # Fbeta_measure = (1 + self.beta ** 2) * tp / ((1 + self.beta ** 2) * tp + fp + self.beta ** 2 * fn)
        # Gbeta_measure = tp / (tp + fp + self.beta * fn)
        # FG_mean = (Fbeta_measure + Gbeta_measure) / 2
        # best_threshold = threshold[np.argmax(FG_mean)]
        #
        # val_pred_label = np.ceil(val_pred_score-best_threshold)
        # TP = np.sum(val_pred_label * val_targ)
        # FP = np.sum(val_pred_label * (1 - val_targ))
        # FN = np.sum((1 - val_pred_label) * val_targ)
        # Fbeta_measure = (1 + self.beta ** 2) * TP / ((1 + self.beta ** 2) * TP + FP + self.beta ** 2 * FN)
        # Gbeta_measure = TP / (TP + FP + self.beta * FN)
        # FG_mean = (Fbeta_measure + Gbeta_measure) / 2
        # print("- val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
        #       % (Fbeta_measure, Gbeta_measure, FG_mean))

        with open(os.path.join(self.save_dir, f"log-epoch{epoch+1:03d}-FG_mean{FG_mean:.3f}.txt"), 'a', encoding='utf-8') as f:
            f.write("val_accuracy:% f \nval_f_measure:% f \nval_Fbeta_measure:% f \nval_Gbeta_measure:% f \nGeometric Mean:% f"
              % (accuracy, f_measure, Fbeta_measure, Gbeta_measure, FG_mean))

lr = 0.001
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr), metrics=['accuracy'])
# model.compile(loss=weighted_mse, optimizer=Adam(lr=lr), metrics=['accuracy'])
# model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr), metrics=['accuracy'])

model_save_dir = make_save_dir(save_dir, "decision_model_final")

stopping = keras.callbacks.EarlyStopping(patience=20)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    factor=0.1,
    patience=5,
    verbose=1,
    mode='min',
    min_lr=lr * 0.01)

checkpointer = keras.callbacks.ModelCheckpoint(
    filepath=get_filename_for_saving(model_save_dir),
    save_best_only=False)

metrics = Metrics(save_dir=model_save_dir)
model.fit(f_train, y_train, epochs=100, batch_size=10, validation_data=(f_dev, y_dev),
          callbacks=[checkpointer, metrics, reduce_lr, stopping])

