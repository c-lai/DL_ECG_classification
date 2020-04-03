from keras import backend as K

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


def build_test_network(**params):
    from keras.models import Sequential, Model
    from keras.layers import SimpleRNN, Dropout, Dense, Input, TimeDistributed, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional
    from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

    inputs = Input(shape=[12, None])

    rnn_layers = []
    for i in range(24):
        split = Lambda(lambda x: x[:, i, :])(inputs)
        reshape = Reshape((-1, params['step']))(split)
        # conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(reshape)
        # BN_layer1 = BatchNormalization()(conv1)
        # relu1 = ReLU()(BN_layer1)
        # dropout1 = Dropout(0.1)(relu1)
        # conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(dropout1)
        # BN_layer2 = BatchNormalization()(conv2)
        # relu2 = ReLU()(BN_layer2)
        # dropout2 = Dropout(0.1)(relu2)

        LSTMlayer1 = Bidirectional(LSTM(64, return_sequences=True))(reshape)
        BN_layer1 = BatchNormalization()(LSTMlayer1)
        LSTMlayer2 = Bidirectional(LSTM(64, return_sequences=True))(BN_layer1)
        BN_layer2 = BatchNormalization()(LSTMlayer2)
        LSTMlayer3 = LSTM(32)(BN_layer2)
        rnn_layers.append(LSTMlayer3)

    merge_layer = concatenate(rnn_layers)

    dropout_layer = Dropout(0.1)(merge_layer)
    output = Dense(9, activation='softmax')(dropout_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam",
                  metrics=["categorical_accuracy"])

    print(model.summary())

    return model


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from evaluate_12ECG_score import compute_beta_score

class Metrics(Callback):
    def __init__(self, val_data, batch_size):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

        self.num_classes = 9
        self.beta = 2

    def on_train_begin(self, logs={}):
        self.val_accuracy = []
        self.val_f_measure = []
        self.val_Fbeta_measure = []
        self.val_Gbeta_measure = []

    def on_epoch_end(self, epoch, logs={}):
        # batches = sum(1 for x in self.validation_data)
        # total = batches * self.batch_size
        #
        # val_pred_score = np.zeros((total, self.num_classes))
        # val_pred_label = np.zeros((total, self.num_classes), dtype=int)
        # val_targ = np.zeros((total, self.num_classes), dtype=int)
        # for batch in range(batches):
        #     xVal, yVal = next(self.validation_data)
        #     val_pred_score[batch * self.batch_size: (batch + 1) * self.batch_size] = np.asarray(self.model.predict(xVal))
        #     val_targ[batch * self.batch_size: (batch + 1) * self.batch_size] = yVal

        # val_pred_score = []
        # val_targ = []
        # for xyVal in enumerate(self.validation_data):
        #     val_pred_score_batch = np.asarray(self.model.predict(xyVal[1][0]))
        #     val_pred_score.append(val_pred_score_batch)
        #     val_targ.append(xyVal[1][1])
        #
        # val_pred_score = np.array(val_pred_score).reshape(-1, self.num_classes)
        # val_targ = np.array(val_targ).reshape(-1, self.num_classes)

        val_pred_score = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]

        val_pred_label = np.zeros((val_pred_score.shape[0], self.num_classes), dtype=int)
        labels = np.argmax(val_pred_score, axis=1)
        for i, label in enumerate(labels):
            val_pred_label[i, label] = 1

        # _val_f1 = f1_score(val_targ, val_predict)
        # _val_recall = recall_score(val_targ, val_predict)
        # _val_precision = precision_score(val_targ, val_predict)
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        accuracy, f_measure, Fbeta_measure, Gbeta_measure = compute_beta_score(val_targ, val_pred_label, self.beta,
                                                                               self.num_classes)
        self.val_accuracy.append(accuracy)
        self.val_f_measure.append(f_measure)
        self.val_Fbeta_measure.append(Fbeta_measure)
        self.val_Gbeta_measure.append(Gbeta_measure)
        print(" - val_accuracy:% f - val_f_measure:% f - val_Fbeta_measure:% f - val_Gbeta_measure:% f"
              % (accuracy, f_measure, Fbeta_measure, Gbeta_measure))
        return


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

    for i in range(24):
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
    LSTM_output = LSTM(32)(relu_output)

    output = Dense(9, activation='softmax')(LSTM_output)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=["categorical_accuracy"])


    print(model.summary())

    return model

