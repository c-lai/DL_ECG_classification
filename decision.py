import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import GradientBoostingClassifier
from evaluate_12ECG_score import compute_beta_score
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import keras
from network import weighted_mse, weighted_cross_entropy
from sklearn.svm import NuSVC

f_1 = loadmat('features_train.mat')
# f_train_1 = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_4_train']), axis=1)
f_2 = loadmat('features_bm_train_standard.mat')
# f_train_2 = np.concatenate((f['features_1_train'], f['features_2_train'], f['features_4_train']), axis=1)
f_train = np.concatenate((f_1['features_1_train'], f_1['features_2_train'], f_1['features_4_train'],
                          f_2['features_bm_1_train_standard'], f_2['features_bm_2_train_standard'][:, 2:],
                          f_2['features_bm_4_train_standard'][:, 2:]), axis=1)
# f_train = f['features_train']
y_train = loadmat('y_train.mat')['y_train']

f_1 = loadmat('features_dev.mat')
# f_dev_1 = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_4_dev']), axis=1)
f_2 = loadmat('features_bm_dev_standard.mat')
# f_dev_2 = np.concatenate((f['features_1_dev'], f['features_2_dev'], f['features_4_dev']), axis=1)
f_dev = np.concatenate((f_1['features_1_dev'][:1375,:], f_1['features_2_dev'][:1375,:], f_1['features_4_dev'][:1375,:],
                        f_2['features_bm_1_dev_standard'], f_2['features_bm_2_dev_standard'][:, 2:],
                        f_2['features_bm_4_dev_standard'][:, 2:]), axis=1)
# f_dev = f['features_dev']
y_dev = loadmat('y_dev.mat')['y_dev']

beta = 2

# decision_model_path = ".\\save\\decision_model\\1586535232-387\\epoch030-val_loss0.815-train_loss0.590.hdf5"
# decision_model = keras.models.load_model(decision_model_path,
#                                           custom_objects={'weighted_mse': weighted_mse,
#                                                           'weighted_cross_entropy': weighted_cross_entropy})
# pred_prob_train = decision_model.predict(f_train)
# pred_prob_dev = decision_model.predict(f_dev)

pred_label_list = []
for i in range(9):
    clf = RandomForestClassifier(n_estimators=150, max_depth=3,
                                    bootstrap=True, random_state=0).fit(f_train, y_train[:, i])
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
    #                                  max_depth=3, random_state=0).fit(f_train, y_train[:, i])
    print(clf.score(f_dev, y_dev[:, i]))

    pred_prob_i_train = clf.predict_proba(f_train)[:, 1]
    # pred_prob_i_train = pred_prob_train[:, i]
    threshold = np.arange(0,1,0.001)
    tn = np.zeros(threshold.shape)
    fp = np.zeros(threshold.shape)
    fn = np.zeros(threshold.shape)
    tp = np.zeros(threshold.shape)
    for n, t in enumerate(threshold):
        tn[n], fp[n], fn[n], tp[n] = confusion_matrix(y_train[:, i], np.ceil(pred_prob_i_train-t)).ravel()
    Fbeta_measure = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
    Gbeta_measure = tp / (tp + fp + beta * fn)
    FG_mean = (Fbeta_measure + Gbeta_measure) / 2
    best_threshold = threshold[np.argmax(FG_mean)]

    pred_prob_i_val = clf.predict_proba(f_dev)[:, 1].reshape((-1, 1))
    # pred_prob_i_val = pred_prob_dev[:, i]
    pred_labels_i = np.ceil(pred_prob_i_val-best_threshold)

    # pred_labels_i = clf.predict(f_dev).reshape((-1, 1))
    pred_label_list.append(pred_labels_i)
    targ_labels_i = y_dev[:, i].reshape((-1, 1))

    TP = np.sum(pred_labels_i*targ_labels_i)
    FP = np.sum(pred_labels_i*(1-targ_labels_i))
    FN = np.sum((1-pred_labels_i)*targ_labels_i)
    Fbeta_measure = (1+beta**2)*TP/((1+beta**2)*TP+FP+beta**2*FN)
    Gbeta_measure = TP/(TP+FP+beta*FN)
    FG_mean = (Fbeta_measure+Gbeta_measure)/2
    print("- val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
        % (Fbeta_measure, Gbeta_measure, FG_mean))

targ_label = y_dev
pred_label = np.concatenate(pred_label_list, axis=1)
accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(targ_label, pred_label, beta, 9)
FG_mean = np.mean([Fbeta_measure, Gbeta_measure])
print("All: - val_accuracy:% f - val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f - Geometric Mean:% f"
      % (accuracy, f_measure, Fbeta_measure, Gbeta_measure, FG_mean))