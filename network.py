from keras import backend as K
from network_util import weighted_mse, weighted_binary_crossentropy, weighted_cross_entropy


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


def build_network_LSTM(**params):
    from keras.models import Model
    from keras.layers import SimpleRNN, Dropout, Dense, Input, TimeDistributed, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional
    from keras.optimizers import Adam

    inputs = Input(shape=[12, None])

    leads_reshape = []
    for i in range(12):
        split = Lambda(lambda x: x[:, i, :])(inputs)
        reshape = Reshape((-1, params['step']))(split)
        leads_reshape.append(reshape)

    input_reshape = concatenate(leads_reshape, axis=-1)
    LSTMlayer1 = Bidirectional(LSTM(100, return_sequences=True))(input_reshape)
    dropout_1 = Dropout(0.5)(LSTMlayer1)
    LSTMlayer2 = Bidirectional(LSTM(100))(dropout_1)
    dropout_2 = Dropout(0.5)(LSTMlayer2)

    # merge_layer = concatenate(rnn_layers)
    output = Dense(9, activation='softmax')(dropout_2)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    return model


def build_network_ResNet(**params):
    from keras.models import Model
    from keras.layers import Dropout, Dense, Input, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional, MaxPooling1D, Add
    from keras.optimizers import Adam, RMSprop
    from keras.initializers import glorot_normal

    inputs = Input(shape=[1, None])

    input_reshape = Reshape((-1, 1))(inputs)
    conv_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(input_reshape)
    bn_1 = BatchNormalization()(conv_1)
    relu_1 = ReLU()(bn_1)
    block_end_1 = relu_1

    conv_2_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(block_end_1)
    bn_2 = BatchNormalization()(conv_2_1)
    relu_2 = ReLU()(bn_2)
    dropout_2 = Dropout(0.3)(relu_2)
    conv_2_2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(dropout_2)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(block_end_1)
    block_end_2 = Add()([conv_2_2, max_pool_2])

    block_end = block_end_2

    # ResNet blocks
    for i in range(params["block_num"]):
        bn_block_1 = BatchNormalization()(block_end)
        relu_block_1 = ReLU()(bn_block_1)
        conv_block_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(relu_block_1)
        bn_block_2 = BatchNormalization()(conv_block_1)
        relu_block_2 = ReLU()(bn_block_2)
        # dropout_block = Dropout(0.3)(relu_block_2)
        conv_block_2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same')(relu_block_2)
        max_pool_block = MaxPooling1D(pool_size=2, strides=2, padding='same')(block_end)
        block_end = Add()([conv_block_2, max_pool_block])

    bn_output_1 = BatchNormalization()(block_end)
    relu_output_1 = ReLU()(bn_output_1)
    LSTM_output = LSTM(32)(relu_output_1)
    bn_output_2 = BatchNormalization()(LSTM_output)
    relu_output_2 = ReLU()(bn_output_2)

    output = Dense(9, activation='sigmoid')(relu_output_2)

    model = Model(inputs=inputs, outputs=output)

    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    print(model.summary())

    return model


def build_network_ResNet_TimeDistributed(**params):
    from keras.models import Model
    from keras.layers import Dropout, Dense, Input, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional, MaxPooling1D, Add
    from keras.optimizers import Adam, RMSprop
    from keras.initializers import glorot_normal

    inputs = Input(shape=[1, None])

    input_reshape = Reshape((-1, params['step'], 1))(inputs)
    conv_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(input_reshape)
    bn_1 = BatchNormalization()(conv_1)
    relu_1 = ReLU()(bn_1)
    block_end_1 = relu_1

    conv_2_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(block_end_1)
    bn_2 = BatchNormalization()(conv_2_1)
    relu_2 = ReLU()(bn_2)
    dropout_2 = Dropout(0.5)(relu_2)
    conv_2_2 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=2, padding='same'))(dropout_2)
    max_pool_2 = TimeDistributed(MaxPooling1D(pool_size=2, strides=2, padding='same'))(block_end_1)
    block_end_2 = Add()([conv_2_2, max_pool_2])

    block_end = block_end_2

    # ResNet blocks
    for i in range(params["block_num"]):
        bn_block_1 = BatchNormalization()(block_end)
        relu_block_1 = ReLU()(bn_block_1)
        conv_block_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(relu_block_1)
        bn_block_2 = BatchNormalization()(conv_block_1)
        relu_block_2 = ReLU()(bn_block_2)
        dropout_block = Dropout(0.5)(relu_block_2)
        conv_block_2 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=2, padding='same'))(dropout_block)
        max_pool_block = TimeDistributed(MaxPooling1D(pool_size=2, strides=2, padding='same'))(block_end)
        block_end = Add()([conv_block_2, max_pool_block])

    bn_output_1 = BatchNormalization()(block_end)
    relu_output_1 = ReLU()(bn_output_1)
    reshape_output = Reshape((-1, int(32 * params['step']/(2**(params["block_num"]+1)))))(relu_output_1)
    LSTM_output = LSTM(32)(reshape_output)
    bn_output_2 = BatchNormalization()(LSTM_output)
    relu_output_2 = ReLU()(bn_output_2)

    output = Dense(9, activation='sigmoid')(relu_output_2)

    model = Model(inputs=inputs, outputs=output)

    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    print(model.summary())

    return model


def build_network_ResNet_12leads(**params):
    from keras.models import Model
    from keras.layers import Dropout, Dense, Input, \
        Lambda, concatenate, Reshape, LSTM, TimeDistributed, Conv1D, \
        BatchNormalization, ReLU, Bidirectional, MaxPooling1D, Add
    from keras.optimizers import Adam, RMSprop
    from keras.initializers import glorot_normal

    leads_num = 12
    inputs = Input(shape=[leads_num, None])

    leads_networks = []
    for i in range(leads_num):
        split = Lambda(lambda x: x[:, i, :])(inputs)
        conv_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(split)
        bn_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(bn_1)
        block_end_1 = relu_1

        conv_2_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(block_end_1)
        bn_2 = BatchNormalization()(conv_2_1)
        relu_2 = ReLU()(bn_2)
        dropout_2 = Dropout(0.5)(relu_2)
        conv_2_2 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=2, padding='same'))(dropout_2)
        max_pool_2 = TimeDistributed(MaxPooling1D(pool_size=2, strides=2, padding='same'))(block_end_1)
        block_end_2 = Add()([conv_2_2, max_pool_2])

        block_end = block_end_2

        # ResNet blocks
        for i in range(params["block_num"]):
            bn_block_1 = BatchNormalization()(block_end)
            relu_block_1 = ReLU()(bn_block_1)
            conv_block_1 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))(relu_block_1)
            bn_block_2 = BatchNormalization()(conv_block_1)
            relu_block_2 = ReLU()(bn_block_2)
            dropout_block = Dropout(0.5)(relu_block_2)
            conv_block_2 = TimeDistributed(Conv1D(filters=32, kernel_size=5, strides=2, padding='same'))(dropout_block)
            max_pool_block = TimeDistributed(MaxPooling1D(pool_size=2, strides=2, padding='same'))(block_end)
            block_end = Add()([conv_block_2, max_pool_block])

        bn_output_1 = BatchNormalization()(block_end)
        relu_output_1 = ReLU()(bn_output_1)
        LSTM_output = LSTM(32)(relu_output_1)
        bn_output_2 = BatchNormalization()(LSTM_output)
        relu_output_2 = ReLU()(bn_output_2)

        leads_networks.append(relu_output_2)

    merge_layer = concatenate(leads_networks)
    dense_1 = Dense(32, activation='relu')(merge_layer)
    dense_2 = Dense(32, activation='relu')(dense_1)
    output = Dense(9, activation='sigmoid')(dense_2)

    model = Model(inputs=inputs, outputs=output)

    optimizer = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))

    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    print(model.summary())

    return model

