# Use single lead models to extract features on external test data
# Instruction: change data_path_ext, leads_model_path, and save file names

import numpy as np
import keras
import os

import load
from network_util import weighted_mse, weighted_cross_entropy, weighted_binary_crossentropy
from scipy.io import savemat
from load import crop


class Preproc_ext:
    def __init__(self, ecg, labels, lead):
        # SNOMED-CT codes
        # (https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv)
        self.labels = ("164889003", "270492004", "164909002", "426783006", "59118001", "284470004",
                       "164884008", "429622005", "164931005",
                       "164930006", "55930002", "6374002", "233917008", "195080001")
        self.choose_label = range(len(self.labels))
        self.choose_leads = lead

    def process(self, x, y):
        # single lead
        return self.process_x(x)[:, self.choose_leads, :], self.process_y(y)[:, self.choose_label]

    def process_x(self, x):
        x_cropped = crop(x)
        x_array = np.asarray(x_cropped, dtype=np.float32)
        return x_array

    def process_y(self, y):
        y_vector = np.full((len(y), 14), 0)
        for i, label in enumerate(y):
            for j, ref in enumerate(self.labels):
                if ref in label:
                    y_vector[i, j] = 1

        return y_vector

# path for data
data_path_ext = ".\\data\\test_E"

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

# load data
dataset_ext = load.load_dataset(data_path_ext, False)

# load models and calculate features
features_ext_test = []
for i, lead in enumerate(leads):
    # load preprocessor and model
    preproc_i = Preproc_ext(*dataset_ext, [lead-1])
    # preproc_i = util.load(os.path.dirname(leads_model_path['lead'+str(lead)]))
    model_i = keras.models.load_model(leads_model_path['lead'+str(lead)],
                                      custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    feature_model_i = keras.Model(inputs=model_i.input, outputs=model_i.layers[-3].output)

    # calculate features for test set
    test_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset_ext)
    y_ext_i = np.empty((len(dataset_ext[0]), 14), dtype=np.int64)
    for n in range(int(len(dataset_ext[0]) / batch_size)):
        y_ext_i[n, :] = next(test_gen_i)[1]
    features_ext_i = feature_model_i.predict(test_gen_i, steps=int(len(dataset_ext[0]) / batch_size), verbose=1)
    features_ext_test.append(features_ext_i)

# save data
if not os.path.exists('.\\features'):
    os.makedirs('.\\features')

savemat('.\\features\\features_external_test_E.mat', {'features_1_test': features_ext_test[0],
                              'features_2_test': features_ext_test[1],
                              'features_3_test': features_ext_test[2],
                              'features_4_test': features_ext_test[3],
                              'features_5_test': features_ext_test[4],
                              'features_6_test': features_ext_test[5],
                              'features_7_test': features_ext_test[6],
                              'features_8_test': features_ext_test[7],
                              'features_9_test': features_ext_test[8],
                              'features_10_test': features_ext_test[9],
                              'features_11_test': features_ext_test[10],
                              'features_12_test': features_ext_test[11]})


savemat('.\\features\\y_external_test_E.mat', {'y_test': y_ext_i})