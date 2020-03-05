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
    # layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    # return Activation('softmax')(layer)
    layer = Dense(params["num_categories"], activation='softmax')(layer)
    return layer


def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

def build_network(**params):
    from keras.models import Model
    from keras.layers import Input, Lambda, Reshape, concatenate
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    # layers = []
    # for i in range(12):
    #     split = Lambda(lambda x: x[:, i, :])(inputs)
    #     reshape = Reshape((-1, params['step']))(split)
    #     if params.get('is_regular_conv', False):
    #         layer = add_conv_layers(reshape, **params)
    #     else:
    #         layer = add_resnet_layers(reshape, **params)
    #     layers.append(layer)
    #
    # merge_layer = concatenate(layers)
    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)

    print(model.summary())

    return model


def build_test_network(**params):
    from keras.models import Sequential, Model
    from keras.layers import SimpleRNN, Dropout, Dense, Input, TimeDistributed, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional
    from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

    inputs = Input(shape=[12, None])

    rnn_layers = []
    for i in range(12):
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

