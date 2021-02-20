import argparse
import numpy as np
import keras
import os
import json
from scipy.io import savemat

import load
import util
from network_util import weighted_binary_crossentropy
from evaluate_12ECG_score import compute_beta_score

batch_size = 1

def predict(data_path, model_config_file):
    dataset = load.load_dataset(data_path, False)

    model_config = json.load(open(model_config_file, 'r'))
    with open(model_config['subset_path']) as f:
        lead_subset = json.load(f)[0]

    print("Loading models and extracting features...")
    # extract features from single-lead ECG
    features = []
    for i, lead in enumerate(lead_subset):
        # load preprocessor and model
        preproc_i = util.load(os.path.dirname(model_config['lead' + str(lead+1)]))
        model_i = keras.models.load_model(model_config['lead' + str(lead+1)],
                                          custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
        feature_model_i = keras.Model(inputs=model_i.input, outputs=model_i.layers[-3].output)

        data_gen_i = load.data_generator_no_shuffle(batch_size, preproc_i, *dataset)

        y_i = np.empty((len(dataset[0]), 9), dtype=np.int64)
        for n in range(int(len(dataset[0]) / batch_size)):
            y_i[n, :] = next(data_gen_i)[1]

        # extract features
        features_i = feature_model_i.predict(data_gen_i, steps=int(len(dataset[0]) / batch_size), verbose=1)
        features.append(features_i)

    features_array = np.concatenate(features, axis=1)
    # load model for final decision making
    print("Making decisions...")
    decision_model = keras.models.load_model(model_config['decision_model'])

    # predict ECG label
    pred_score = np.asarray(decision_model.predict(features_array))
    pred_label = np.ceil(pred_score - 0.5).astype('int32')
    true_label = y_i

    accuracy, f1, Fbeta, Gbeta = \
        compute_beta_score(true_label, pred_label, 1, 9)
    metrics = {"accuracy": accuracy,
               "f1": f1}
    print(f'accuracy:{accuracy:.3f}, f1:{f1:.3f}')

    return pred_score, pred_label, true_label, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data")
    parser.add_argument("model_config_file", help="path to config file of model")
    args = parser.parse_args()
    pred_score, pred_label, true_label, metrics = predict(args.data_path, args.model_config_file)
    print("Saving result...")
    savemat(".\\predict_result.mat",
            {
                "pred_score": pred_score,
                "pred_label": pred_label,
                "true_label": true_label,
                "metrics": metrics
            })
