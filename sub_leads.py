import numpy as np
import keras
import os

import load
import util
from keras import Model
from network import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from scipy.io import savemat
from get_12ECG_features import get_12ECG_features


def get_12ECG_features_all(data, header_data, model):
    preproc = util.load(os.path.dirname(leads_model_path['lead1']))
    dataset = (np.expand_dims(data, axis=0), header_data[15][5:-1])

    features_NN = []
    for i, lead in enumerate(leads):
        dataset_leadi = ([dataset[0][:, lead-1, :]], [dataset[1]])
        data_x, data_y = preproc.process(*dataset_leadi)
        feature_model = keras.Model(inputs=model[i].input, outputs=model[i].layers[-3].output)
        feature_NN_i = feature_model.predict(data_x, verbose=0)
        features_NN.append(feature_NN_i)

    features_benchmark = []
    for l in leads:
        feature_benchmark_l = np.asarray(get_12ECG_features(data, header_data, l))
        feats_reshape_l = feature_benchmark_l.reshape(1, -1)
        features_benchmark.append(feats_reshape_l)

    features = np.concatenate([np.concatenate(features_NN, axis=1), np.concatenate(features_benchmark, axis=1)], axis=1)

    return features, data_y

# path for data
data_path_train = ".\\Training_WFDB\\train"
data_path_dev = ".\\Training_WFDB\\dev"
data_path_all = ".\\Training_WFDB\\all"
data_path_test = ".\\Training_WFDB\\test_set"

# path for lead models
leads = range(1, 13)
# leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE\\epoch022-val_loss0.393-train_loss0.316.hdf5",
#                     'lead2': ".\\save\\lead2_ResNet8_32_WCE\\epoch045-val_loss0.217-train_loss0.247.hdf5",
#                     'lead3': ".\\save\\lead3_ResNet8_32_WCE\\epoch024-val_loss0.291-train_loss0.329.hdf5",
#                     'lead4': ".\\save\\lead4_ResNet8_32_WCE\\epoch029-val_loss0.194-train_loss0.208.hdf5",
#                     'lead5': ".\\save\\lead5_ResNet8_32_WCE\\epoch030-val_loss0.354-train_loss0.319.hdf5",
#                     'lead6': ".\\save\\lead6_ResNet8_32_WCE\\epoch025-val_loss0.331-train_loss0.322.hdf5",
#                     'lead7': ".\\save\\lead7_ResNet8_32_WCE\\epoch030-val_loss0.315-train_loss0.298.hdf5",
#                     'lead8': ".\\save\\lead8_ResNet8_32_WCE\\epoch024-val_loss0.614-train_loss0.318.hdf5",
#                     'lead9': ".\\save\\lead9_ResNet8_32_WCE\\epoch023-val_loss0.278-train_loss0.287.hdf5",
#                     'lead10': ".\\save\\lead10_ResNet8_32_WCE\\epoch039-val_loss0.307-train_loss0.282.hdf5",
#                     'lead11': ".\\save\\lead11_ResNet8_32_WCE\\epoch027-val_loss0.485-train_loss0.320.hdf5",
#                     'lead12': ".\\save\\lead12_ResNet8_32_WCE\\epoch018-val_loss0.843-train_loss0.339.hdf5"}
leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.557-train_loss0.156.hdf5",
                    'lead2': ".\\save\\lead2_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.241-train_loss0.089.hdf5",
                    'lead3': ".\\save\\lead3_ResNet8_NoDropout_32_WCE\\epoch022-val_loss0.279-train_loss0.124.hdf5",
                    'lead4': ".\\save\\lead4_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.300-train_loss0.130.hdf5",
                    'lead5': ".\\save\\lead5_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.389-train_loss0.206.hdf5",
                    'lead6': ".\\save\\lead6_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.275-train_loss0.129.hdf5",
                    'lead7': ".\\save\\lead7_ResNet8_NoDropout_32_WCE\\epoch008-val_loss0.738-train_loss0.226.hdf5",
                    'lead8': ".\\save\\lead8_ResNet8_NoDropout_32_WCE\\epoch015-val_loss0.321-train_loss0.166.hdf5",
                    'lead9': ".\\save\\lead9_ResNet8_NoDropout_32_WCE\\epoch052-val_loss0.307-train_loss0.183.hdf5",
                    'lead10': ".\\save\\lead10_ResNet8_NoDropout_32_WCE\\epoch024-val_loss0.367-train_loss0.146.hdf5",
                    'lead11': ".\\save\\lead11_ResNet8_NoDropout_32_WCE\\epoch020-val_loss0.494-train_loss0.175.hdf5",
                    'lead12': ".\\save\\lead12_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.939-train_loss0.191.hdf5"}

# load data
dataset_dev = load.load_dataset(data_path_dev, False)
dataset_train = load.load_dataset(data_path_train, False)
# dataset_all = load.load_dataset(data_path_all, False)
dataset_test = load.load_testset(data_path_test)

# load models and calculate features
features_dev = []
features_train = []
features_all = []
features_test = []
for i, lead in enumerate(leads):
    preproc_i = util.load(os.path.dirname(leads_model_path['lead'+str(lead)]))
    model_i = keras.models.load_model(leads_model_path['lead'+str(lead)],
                                      custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    feature_model_i = Model(inputs=model_i.input, outputs=model_i.layers[-3].output)

    x_dev_i, y_dev_i = preproc_i.process(*dataset_dev)
    features_dev_i = feature_model_i.predict(x_dev_i, verbose=1)
    features_dev.append(features_dev_i)

    x_train_i, y_train_i = preproc_i.process(*dataset_train)
    features_train_i = feature_model_i.predict(x_train_i, verbose=1)
    features_train.append(features_train_i)

    # x_all_i, y_all_i = preproc_i.process(*dataset_all)
    # features_all_i = feature_model_i.predict(x_all_i, verbose=1)
    # features_all.append(features_all_i)

    x_test_i, y_test_i = preproc_i.process(*dataset_test)
    features_test_i = feature_model_i.predict(x_test_i, verbose=1)
    features_test.append(features_test_i)

# save data
savemat('features_NoDropout_dev.mat', {'features_1_dev': features_dev[0],
                             'features_2_dev': features_dev[1],
                             'features_3_dev': features_dev[2],
                             'features_4_dev': features_dev[3],
                             'features_5_dev': features_dev[4],
                             'features_6_dev': features_dev[5],
                             'features_7_dev': features_dev[6],
                             'features_8_dev': features_dev[7],
                             'features_9_dev': features_dev[8],
                             'features_10_dev': features_dev[9],
                             'features_11_dev': features_dev[10],
                             'features_12_dev': features_dev[11]})
savemat('features_NoDropout_train.mat', {'features_1_train': features_train[0],
                               'features_2_train': features_train[1],
                               'features_3_train': features_train[2],
                               'features_4_train': features_train[3],
                               'features_5_train': features_train[4],
                               'features_6_train': features_train[5],
                               'features_7_train': features_train[6],
                               'features_8_train': features_train[7],
                               'features_9_train': features_train[8],
                               'features_10_train': features_train[9],
                               'features_11_train': features_train[10],
                               'features_12_train': features_train[11]})
# savemat('features_all.mat', {'features_1_all': features_all[0],
#                              'features_2_all': features_all[1],
#                              'features_3_all': features_all[2],
#                              'features_4_all': features_all[3],
#                              'features_5_all': features_all[4],
#                              'features_6_all': features_all[5],
#                              'features_7_all': features_all[6],
#                              'features_8_all': features_all[7],
#                              'features_9_all': features_all[8],
#                              'features_10_all': features_all[9],
#                              'features_11_all': features_all[10],
#                              'features_12_all': features_all[11]})
savemat('features_NoDropout_test.mat', {'features_1_test': features_test[0],
                              'features_2_test': features_test[1],
                              'features_3_test': features_test[2],
                              'features_4_test': features_test[3],
                              'features_5_test': features_test[4],
                              'features_6_test': features_test[5],
                              'features_7_test': features_test[6],
                              'features_8_test': features_test[7],
                              'features_9_test': features_test[8],
                              'features_10_test': features_test[9],
                              'features_11_test': features_test[10],
                              'features_12_test': features_test[11]})

savemat('y_NoDropout_dev.mat', {'y_dev': y_dev_i})
savemat('y_NoDropout_train.mat', {'y_train': y_train_i})
# savemat('y_all.mat', {'y_all': y_all_i})
savemat('y_NoDropout_test.mat', {'y_test': y_test_i})