from __future__ import print_function

import argparse
import numpy as np
import keras
import tensorflow as tf
import os

import load
import util
from keras import Model
from network import weighted_mse, weighted_cross_entropy
from scipy.io import savemat
from get_12ECG_features import get_12ECG_features
from scipy.io import loadmat

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def save_challenge_predictions(output_directory, filename, scores, labels, classes):
    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat', '.csv')
    output_file = os.path.join(output_directory, new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


# Find unique number of classes
def get_classes(input_directory, files):
    classes = set()
    for f in files:
        g = f.replace('.mat', '.hea')
        input_file = os.path.join(input_directory, g)
        with open(input_file, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_path_train = ".\\Training_WFDB\\train"
data_path_dev = ".\\Training_WFDB\\dev"
data_path_all = ".\\Training_WFDB\\all"


# lead1_model_path = ".\\save\\lead1_final\\epoch014-val_loss0.442-train_loss0.197.hdf5"
# lead_1 = 1
#
# lead2_model_path = ".\\save\\lead2_final\\epoch020-val_loss0.044-train_loss0.109.hdf5"
# lead_2 = 2
#
# lead4_model_path = ".\\save\\lead4_final\\epoch027-val_loss0.380-train_loss0.208.hdf5"
# lead_4 = 4

# model_path_5 = ".\\save\\lead1_ResNet16_WMSE\\1586231660-902\\epoch030-val_loss0.393-train_loss0.236.hdf5"
# lead_5 = 5
#
# model_path_6 = ".\\save\\lead1_ResNet16_WMSE\\1586231660-902\\epoch030-val_loss0.393-train_loss0.236.hdf5"
# lead_6 = 6

leads = [1, 2, 4]
lead1_model_path = ".\\save\\lead1_final\\epoch014-val_loss0.442-train_loss0.197.hdf5"
lead2_model_path = ".\\save\\lead2_final\\epoch020-val_loss0.044-train_loss0.109.hdf5"
lead4_model_path = ".\\save\\lead4_final\\epoch027-val_loss0.380-train_loss0.208.hdf5"
# lead5_model_path = ".\\save\\lead1_ResNet8_64_WMSE\\1586365571-169\\epoch014-val_loss0.131-train_loss0.197.hdf5"
# lead6_model_path = ".\\save\\lead1_ResNet8_64_WMSE\\1586365571-169\\epoch014-val_loss0.131-train_loss0.197.hdf5"
# final_model_path = ".\\save\\lead1_ResNet8_64_WMSE\\1586365571-169\\epoch014-val_loss0.131-train_loss0.197.hdf5"


def get_12ECG_features_all(data, header_data, model):
    preproc = util.load(os.path.dirname(lead1_model_path))
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

#############################################################################
preproc_1 = util.load(os.path.dirname(lead1_model_path))
model_1 = keras.models.load_model(lead1_model_path,
                                custom_objects=
                                {'weighted_mse': weighted_mse, 'weighted_cross_entropy': weighted_cross_entropy})
feature_model_1 = Model(inputs=model_1.input, outputs=model_1.layers[-4].output)
dataset_1_dev = load.load_dataset(data_path_dev, leads[0])
x_1_dev, y_1_dev = preproc_1.process(*dataset_1_dev)
features_1_dev = feature_model_1.predict(x_1_dev, verbose=1)

dataset_1_train = load.load_dataset(data_path_train, leads[0])
x_1_train, y_1_train = preproc_1.process(*dataset_1_train)
features_1_train = feature_model_1.predict(x_1_train, verbose=1)

dataset_1_all = load.load_dataset(data_path_all, leads[0])
x_1_all, y_1_all = preproc_1.process(*dataset_1_all)
features_1_all = feature_model_1.predict(x_1_all, verbose=1)


preproc_2 = util.load(os.path.dirname(lead2_model_path))
model_2 = keras.models.load_model(lead2_model_path,
                                custom_objects=
                                {'weighted_mse': weighted_mse, 'weighted_cross_entropy': weighted_cross_entropy})
feature_model_2 = Model(inputs=model_2.input, outputs=model_2.layers[-4].output)
dataset_2_dev = load.load_dataset(data_path_dev, leads[1])
x_2_dev, y_2_dev = preproc_2.process(*dataset_2_dev)
features_2_dev = feature_model_2.predict(x_2_dev, verbose=1)

dataset_2_train = load.load_dataset(data_path_train, leads[1])
x_2_train, y_2_train = preproc_2.process(*dataset_2_train)
features_2_train = feature_model_2.predict(x_2_train, verbose=1)

dataset_2_all = load.load_dataset(data_path_all, leads[1])
x_2_all, y_2_all = preproc_2.process(*dataset_2_all)
features_2_all = feature_model_2.predict(x_2_all, verbose=1)


preproc_4 = util.load(os.path.dirname(lead4_model_path))
model_4 = keras.models.load_model(lead4_model_path,
                                custom_objects=
                                {'weighted_mse': weighted_mse, 'weighted_cross_entropy': weighted_cross_entropy})
feature_model_4 = Model(inputs=model_4.input, outputs=model_4.layers[-4].output)
dataset_4_dev = load.load_dataset(data_path_dev, leads[2])
x_4_dev, y_4_dev = preproc_4.process(*dataset_4_dev)
features_4_dev = feature_model_4.predict(x_4_dev, verbose=1)

dataset_4_train = load.load_dataset(data_path_train, leads[2])
x_4_train, y_4_train = preproc_4.process(*dataset_4_train)
features_4_train = feature_model_4.predict(x_4_train, verbose=1)

dataset_4_all = load.load_dataset(data_path_all, leads[2])
x_4_all, y_4_all = preproc_4.process(*dataset_4_all)
features_4_all = feature_model_4.predict(x_4_all, verbose=1)


# model_5 = keras.models.load_model(model_path_5,
#                                 custom_objects=
#                                 {'weighted_mse': weighted_mse, 'weighted_cross_entropy': weighted_cross_entropy})
# feature_model_5 = Model(inputs=model_5.input, outputs=model_5.layers[-2].output)
# dataset_5_dev = load.load_dataset(data_path_dev, lead_5)
# x_5_dev, y_5_dev = preproc.process(*dataset_5_dev)
# features_5_dev = feature_model_5.predict(x_5_dev, verbose=1)
#
# dataset_5_train = load.load_dataset(data_path_train, lead_5)
# x_5_train, y_5_train = preproc.process(*dataset_5_train)
# features_5_train = feature_model_5.predict(x_5_train, verbose=1)
#
#
# model_6 = keras.models.load_model(model_path_6,
#                                 custom_objects=
#                                 {'weighted_mse': weighted_mse, 'weighted_cross_entropy': weighted_cross_entropy})
# feature_model_6 = Model(inputs=model_6.input, outputs=model_6.layers[-2].output)
# dataset_6_dev = load.load_dataset(data_path_dev, lead_6)
# x_6_dev, y_6_dev = preproc.process(*dataset_6_dev)
# features_6_dev = feature_model_6.predict(x_6_dev, verbose=1)
#
# dataset_6_train = load.load_dataset(data_path_train, lead_6)
# x_6_train, y_6_train = preproc.process(*dataset_6_train)
# features_6_train = feature_model_6.predict(x_6_train, verbose=1)

####################################################################
# lead1_model = keras.models.load_model(lead1_model_path,
#                                       custom_objects={'weighted_mse': weighted_mse,
#                                                       'weighted_cross_entropy': weighted_cross_entropy})
# lead2_model = keras.models.load_model(lead2_model_path,
#                                       custom_objects={'weighted_mse': weighted_mse,
#                                                       'weighted_cross_entropy': weighted_cross_entropy})
# lead4_model = keras.models.load_model(lead4_model_path,
#                                       custom_objects={'weighted_mse': weighted_mse,
#                                                       'weighted_cross_entropy': weighted_cross_entropy})
# # lead5_model = keras.models.load_model(lead5_model_path,
# #                                       custom_objects={'weighted_mse': weighted_mse,
# #                                                       'weighted_cross_entropy': weighted_cross_entropy})
# # lead6_model = keras.models.load_model(lead6_model_path,
# #                                       custom_objects={'weighted_mse': weighted_mse,
# #                                                       'weighted_cross_entropy': weighted_cross_entropy})
# # final_model = keras.models.load_model(final_model_path,
# #                                       custom_objects={'weighted_mse': weighted_mse,
# #                                                       'weighted_cross_entropy': weighted_cross_entropy})
#
# loaded_models = [lead1_model, lead2_model, lead4_model]

####################################################################
input_directory = data_path_train

# Find files.
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
        input_files.append(f)

classes = get_classes(input_directory, input_files)

# Iterate over files.
print('Extracting 12ECG features...')
num_files = len(input_files)

features_train = []
y_train = []
features_bm_1_train = []
features_bm_2_train = []
features_bm_4_train = []
for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i + 1, num_files))
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = load_challenge_data(tmp_input_file)

    # feature_train_i, y_train_i = get_12ECG_features_all(data, header_data, loaded_models)
    # features_train.append(feature_train_i)
    # y_train.append(y_train_i)

    features_bm_1_train_i = [get_12ECG_features(data, header_data, 1)]
    features_bm_1_train.append(features_bm_1_train_i)
    features_bm_2_train_i = [get_12ECG_features(data, header_data, 2)]
    features_bm_2_train.append(features_bm_2_train_i)
    features_bm_4_train_i = [get_12ECG_features(data, header_data, 4)]
    features_bm_4_train.append(features_bm_4_train_i)

# features_train_mat = np.concatenate(features_train)
# y_train_mat = np.concatenate(y_train)
features_bm_1_train_mat = np.concatenate(features_bm_1_train)
features_bm_2_train_mat = np.concatenate(features_bm_2_train)
features_bm_4_train_mat = np.concatenate(features_bm_4_train)


####################################################################
input_directory = data_path_dev

# Find files.
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
            'mat'):
        input_files.append(f)

classes = get_classes(input_directory, input_files)

# Iterate over files.
print('Extracting 12ECG features...')
num_files = len(input_files)

features_dev = []
y_dev = []
features_bm_1_dev = []
features_bm_2_dev = []
features_bm_4_dev = []
for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i + 1, num_files))
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = load_challenge_data(tmp_input_file)

    # feature_dev_i, y_dev_i = get_12ECG_features_all(data, header_data, loaded_models)
    # features_dev.append(feature_dev_i)
    # y_dev.append(y_dev_i)

    features_bm_1_dev_i = [get_12ECG_features(data, header_data, 1)]
    features_bm_1_dev.append(features_bm_1_dev_i)
    features_bm_2_dev_i = [get_12ECG_features(data, header_data, 2)]
    features_bm_2_dev.append(features_bm_2_dev_i)
    features_bm_4_dev_i = [get_12ECG_features(data, header_data, 4)]
    features_bm_4_dev.append(features_bm_4_dev_i)

# features_dev_mat = np.concatenate(features_dev)
# y_dev_mat = np.concatenate(y_dev)
features_bm_1_dev_mat = np.concatenate(features_bm_1_dev)
features_bm_2_dev_mat = np.concatenate(features_bm_2_dev)
features_bm_4_dev_mat = np.concatenate(features_bm_4_dev)

####################################################################
input_directory = data_path_all

# Find files.
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
            'mat'):
        input_files.append(f)

classes = get_classes(input_directory, input_files)

# Iterate over files.
print('Extracting 12ECG features...')
num_files = len(input_files)

features_all = []
y_all = []
features_bm_1_all = []
features_bm_2_all = []
features_bm_4_all = []
for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i + 1, num_files))
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = load_challenge_data(tmp_input_file)

    # feature_all_i, y_all_i = get_12ECG_features_all(data, header_data, loaded_models)
    # features_all.append(feature_all_i)
    # y_all.append(y_all_i)

    features_bm_1_all_i = [get_12ECG_features(data, header_data, 1)]
    features_bm_1_all.append(features_bm_1_all_i)
    features_bm_2_all_i = [get_12ECG_features(data, header_data, 2)]
    features_bm_2_all.append(features_bm_2_all_i)
    features_bm_4_all_i = [get_12ECG_features(data, header_data, 4)]
    features_bm_4_all.append(features_bm_4_all_i)

# features_all_mat = np.concatenate(features_all)
# y_all_mat = np.concatenate(y_all)
features_bm_1_all_mat = np.concatenate(features_bm_1_all)
features_bm_2_all_mat = np.concatenate(features_bm_2_all)
features_bm_4_all_mat = np.concatenate(features_bm_4_all)

####################################################################
savemat('features_dev_final.mat', {'features_1_dev':features_1_dev, 'features_2_dev':features_2_dev,
                             'features_4_dev':features_4_dev})
savemat('features_train_final.mat', {'features_1_train':features_1_train, 'features_2_train':features_2_train,
                               'features_4_train':features_4_train})
savemat('features_all_final.mat', {'features_1_all':features_1_all, 'features_2_all':features_2_all,
                               'features_4_all':features_4_all})
savemat('features_bm_dev.mat', {'features_bm_1_dev':features_bm_1_dev_mat, 'features_bm_2_dev':features_bm_2_dev_mat,
                                'features_bm_4_dev':features_bm_4_dev_mat})
savemat('features_bm_train.mat', {'features_bm_1_train':features_bm_1_train_mat, 'features_bm_2_train':features_bm_2_train_mat,
                                  'features_bm_4_train':features_bm_4_train_mat})
savemat('features_bm_all.mat', {'features_bm_1_all':features_bm_1_all_mat, 'features_bm_2_all':features_bm_2_all_mat,
                                  'features_bm_4_all':features_bm_4_all_mat})
# savemat('y_dev_32.mat', {'y_dev':y_1_dev})
# savemat('y_train_32.mat', {'y_train':y_1_train})

# savemat('features_dev_norelu.mat', {'features_dev':features_dev_mat})
# savemat('features_train_norelu.mat', {'features_train':features_train_mat})
savemat('y_dev_final.mat', {'y_dev':y_1_dev})
savemat('y_train_final.mat', {'y_train':y_1_train})
savemat('y_all_final.mat', {'y_all':y_1_all})

