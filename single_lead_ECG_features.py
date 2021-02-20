# Use single lead models to extract features on CPSC dataset (Training_WFDB)
# Instruction: change leads_model_path, and save file names

import numpy as np
import keras
import os

import load
import util
from network_util import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from scipy.io import savemat

# path for data
data_path_train = ".\\data_old\\train"
data_path_dev = ".\\data_old\\dev"
data_path_test = ".\\data_old\\test"

# path for lead models
leads = range(1, 13)
batch_size = 1
leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE\\epoch022-val_loss0.393-train_loss0.316.hdf5",
                    'lead2': ".\\save\\lead2_ResNet8_32_WCE\\epoch045-val_loss0.217-train_loss0.247.hdf5",
                    'lead3': ".\\save\\lead3_ResNet8_32_WCE\\epoch024-val_loss0.291-train_loss0.329.hdf5",
                    'lead4': ".\\save\\lead4_ResNet8_32_WCE\\epoch029-val_loss0.194-train_loss0.208.hdf5",
                    'lead5': ".\\save\\lead5_ResNet8_32_WCE\\epoch030-val_loss0.354-train_loss0.319.hdf5",
                    'lead6': ".\\save\\lead6_ResNet8_32_WCE\\epoch025-val_loss0.331-train_loss0.322.hdf5",
                    'lead7': ".\\save\\lead7_ResNet8_32_WCE\\epoch030-val_loss0.315-train_loss0.298.hdf5",
                    'lead8': ".\\save\\lead8_ResNet8_32_WCE\\epoch024-val_loss0.614-train_loss0.318.hdf5",
                    'lead9': ".\\save\\lead9_ResNet8_32_WCE\\epoch023-val_loss0.278-train_loss0.287.hdf5",
                    'lead10': ".\\save\\lead10_ResNet8_32_WCE\\epoch039-val_loss0.307-train_loss0.282.hdf5",
                    'lead11': ".\\save\\lead11_ResNet8_32_WCE\\epoch027-val_loss0.485-train_loss0.320.hdf5",
                    'lead12': ".\\save\\lead12_ResNet8_32_WCE\\epoch018-val_loss0.843-train_loss0.339.hdf5"}
# leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.557-train_loss0.156.hdf5",
#                     'lead2': ".\\save\\lead2_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.241-train_loss0.089.hdf5",
#                     'lead3': ".\\save\\lead3_ResNet8_NoDropout_32_WCE\\epoch022-val_loss0.279-train_loss0.124.hdf5",
#                     'lead4': ".\\save\\lead4_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.300-train_loss0.130.hdf5",
#                     'lead5': ".\\save\\lead5_ResNet8_NoDropout_32_WCE\\epoch012-val_loss0.389-train_loss0.206.hdf5",
#                     'lead6': ".\\save\\lead6_ResNet8_NoDropout_32_WCE\\epoch014-val_loss0.275-train_loss0.129.hdf5",
#                     'lead7': ".\\save\\lead7_ResNet8_NoDropout_32_WCE\\epoch008-val_loss0.738-train_loss0.226.hdf5",
#                     'lead8': ".\\save\\lead8_ResNet8_NoDropout_32_WCE\\epoch015-val_loss0.321-train_loss0.166.hdf5",
#                     'lead9': ".\\save\\lead9_ResNet8_NoDropout_32_WCE\\epoch052-val_loss0.307-train_loss0.183.hdf5",
#                     'lead10': ".\\save\\lead10_ResNet8_NoDropout_32_WCE\\epoch024-val_loss0.367-train_loss0.146.hdf5",
#                     'lead11': ".\\save\\lead11_ResNet8_NoDropout_32_WCE\\epoch020-val_loss0.494-train_loss0.175.hdf5",
#                     'lead12': ".\\save\\lead12_ResNet8_NoDropout_32_WCE\\epoch017-val_loss0.939-train_loss0.191.hdf5"}
# leads_model_path = {'lead1': ".\\save\\lead1_ResNet8_32_WCE_v2\\epoch032-val_loss0.452-train_loss0.176.hdf5",
#                     'lead2': ".\\save\\lead2_ResNet8_32_WCE_v2\\epoch049-val_loss0.016-train_loss0.153.hdf5",
#                     'lead3': ".\\save\\lead3_ResNet8_32_WCE_v2\\epoch023-val_loss0.095-train_loss0.295.hdf5",
#                     'lead4': ".\\save\\lead4_ResNet8_32_WCE_v2\\epoch076-val_loss0.033-train_loss0.187.hdf5",
#                     'lead5': ".\\save\\lead5_ResNet8_32_WCE_v2\\epoch053-val_loss0.122-train_loss0.197.hdf5",
#                     'lead6': ".\\save\\lead6_ResNet8_32_WCE_v2\\epoch024-val_loss0.068-train_loss0.213.hdf5",
#                     'lead7': ".\\save\\lead7_ResNet8_32_WCE_v2\\epoch021-val_loss0.370-train_loss0.250.hdf5",
#                     'lead8': ".\\save\\lead8_ResNet8_32_WCE_v2\\epoch021-val_loss0.363-train_loss0.277.hdf5",
#                     'lead9': ".\\save\\lead9_ResNet8_32_WCE_v2\\epoch043-val_loss0.015-train_loss0.140.hdf5",
#                     'lead10': ".\\save\\lead10_ResNet8_32_WCE_v2\\epoch100-val_loss0.050-train_loss0.199.hdf5",
#                     'lead11': ".\\save\\lead11_ResNet8_32_WCE_v2\\epoch021-val_loss0.704-train_loss0.260.hdf5",
#                     'lead12': ".\\save\\lead12_ResNet8_32_WCE_v2\\epoch022-val_loss0.452-train_loss0.246.hdf5"}

# load data
dataset_dev = load.load_dataset(data_path_dev, False)
dataset_train = load.load_dataset(data_path_train, False)
dataset_test = load.load_dataset(data_path_test, False)

# load models and calculate features
features_dev = []
features_train = []
features_test = []
for i, lead in enumerate(leads):
    # load preprocessor and model
    preproc_i = util.load(os.path.dirname(leads_model_path['lead'+str(lead)]))
    model_i = keras.models.load_model(leads_model_path['lead'+str(lead)],
                                      custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    feature_model_i = keras.Model(inputs=model_i.input, outputs=model_i.layers[-3].output)

    # calculate features for validation set
    dev_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_dev)
    y_dev_i = np.empty((len(dataset_dev[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_dev[0]) / batch_size)):
        y_dev_i[n, :] = next(dev_gen_i)[1]
    features_dev_i = feature_model_i.predict(dev_gen_i, steps=int(len(dataset_dev[0]) / batch_size), verbose=1)
    features_dev.append(features_dev_i)

    # calculate features for training set
    train_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_train)
    y_train_i = np.empty((len(dataset_train[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_train[0]) / batch_size)):
        y_train_i[n, :] = next(train_gen_i)[1]
    features_train_i = feature_model_i.predict(train_gen_i, steps=int(len(dataset_train[0]) / batch_size), verbose=1)
    features_train.append(features_train_i)

    # calculate features for test set
    test_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_test)
    y_test_i = np.empty((len(dataset_test[0]), 9), dtype=np.int64)
    for n in range(int(len(dataset_test[0]) / batch_size)):
        y_test_i[n, :] = next(test_gen_i)[1]
    features_test_i = feature_model_i.predict(test_gen_i, steps=int(len(dataset_test[0]) / batch_size), verbose=1)
    features_test.append(features_test_i)

# save data
if not os.path.exists('.\\features'):
    os.makedirs('.\\features')

savemat('.\\features\\features_dev_1.mat', {'features_1_dev': features_dev[0],
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
savemat('.\\features\\features_train_1.mat', {'features_1_train': features_train[0],
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
savemat('.\\features\\features_test_1.mat', {'features_1_test': features_test[0],
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

savemat('.\\features\\y_dev.mat', {'y_dev': y_dev_i})
savemat('.\\features\\y_train.mat', {'y_train': y_train_i})
savemat('.\\features\\y_test.mat', {'y_test': y_test_i})