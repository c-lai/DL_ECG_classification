from keras import backend as K
from tensorflow import math as tfmath
import os

def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)
    return model


import numpy as np
from keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from evaluate_12ECG_score import compute_beta_score

class Metrics(Callback):
    def __init__(self, val_data, batch_size, save_dir):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
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

        with open(os.path.join(self.save_dir, f"log-epoch{epoch+1:03d}-FG_mean{FG_mean:.3f}.txt"), 'a', encoding='utf-8') as f:
            f.write("val_accuracy:% f \nval_f_measure:% f \nval_Fbeta_measure:% f \nval_Gbeta_measure:% f \nGeometric Mean:% f"
              % (accuracy, f_measure, Fbeta_measure, Gbeta_measure, FG_mean))

        return


def weighted_mse(yTrue,yPred):
    class_weight = K.constant([[0.043], [0.073], [0.264],
                               [0.057], [0.097], [0.084],
                               [0.031], [0.067], [0.284]])
    class_se = K.square(yPred-yTrue)
    w_mse = K.dot(class_se, class_weight)
    return K.sum(w_mse)


def weighted_cross_entropy(yTrue,yPred):
    class_weight = K.constant([[0.043], [0.073], [0.264],
                               [0.057], [0.097], [0.084],
                               [0.031], [0.067], [0.284]])
    class_CE = -(tfmath.multiply(yTrue, K.log(yPred))+2*tfmath.multiply(1-yTrue, K.log(1-yPred)))
    w_cross_entropy = K.dot(class_CE, class_weight)
    return K.sum(w_cross_entropy)


def build_network_1lead(**params):
    from keras.models import Sequential, Model
    from keras.layers import SimpleRNN, Dropout, Dense, Input, TimeDistributed, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional, MaxPooling1D, Add, LeakyReLU, ELU
    from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
    from keras.optimizers import Adam

    inputs = Input(shape=[1, None])
    reshape = Reshape((-1, 1))(inputs)
    conv = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(reshape)
    BN_layer = BatchNormalization()(conv)
    block_out1 = ReLU()(BN_layer)

    conv1 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(block_out1)
    BN_layer1 = BatchNormalization()(conv1)
    relu1 = ReLU()(BN_layer1)
    dropout1 = Dropout(0.5)(relu1)
    conv2 = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(dropout1)
    max_pool = MaxPooling1D(pool_size=2, padding='same')(block_out1)
    block_out2 = Add()([conv2, max_pool])

    res_block_end = block_out2

    for i in range(8):
        BN_layer_res_1 = BatchNormalization()(res_block_end)
        relu_res_1 = ReLU()(BN_layer_res_1)
        conv_res_1 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(relu_res_1)
        BN_layer_res_2 = BatchNormalization()(conv_res_1)
        relu_res_2 = ReLU()(BN_layer_res_2)
        dropout_res = Dropout(0.5)(relu_res_2)
        conv_res_2 = Conv1D(filters=64, kernel_size=5, strides=2, padding='same')(dropout_res)
        max_pool_res = MaxPooling1D(pool_size=2, padding='same')(res_block_end)
        res_block_end = Add()([conv_res_2, max_pool_res])

    BN_output = BatchNormalization()(res_block_end)
    relu_output = ReLU()(BN_output)
    # LSTM_output_1 = LSTM(32, return_sequences=True)(relu_output)
    # BN_output_1 = BatchNormalization()(LSTM_output_1)
    # relu_output_1 = ReLU()(BN_output_1)
    LSTM_output_2 = LSTM(64)(relu_output)
    BN_output_2 = BatchNormalization()(LSTM_output_2)
    relu_output_2 = ReLU()(BN_output_2)

    output = Dense(9, activation='softmax')(relu_output_2)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss=weighted_mse, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    return model

