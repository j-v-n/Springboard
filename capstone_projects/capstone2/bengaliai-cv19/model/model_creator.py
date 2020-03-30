from keras.models import Model
from keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPool2D,
    Dropout,
    BatchNormalization,
    Input,
)
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.backend import clear_session
from keras import initializers
from keras.callbacks import EarlyStopping


def model_create(
    input_shape=(137, 236, 1),
    n_conv_layers=3,
    dropout_rate1=0.3,
    dense_layer1_count=1024,
    dropout_rate2=0.3,
    dense_layer2_count=512,
    conv_nfilters=100,
    conv_kernel_size=3,
    n_strides=1,
    pool_size=5,
    activation_conv="relu",
    activation_dense="relu",
    activation_out="softmax",
    bn_momentum=0.99,
):

    """
    Function to create a CNN with alternating convolutional and pooling layers,
    followed by a dropout, then 2 dense layers with a dropout in between and finally 3 output layers for the 3 target variables,
    i.e grapheme root, vowel diacritic and consonant diacritic
    
    This uses the Keras Functional API
    """
    clear_session()
    inputs = Input(shape=input_shape, name="input_layer")
    for i in range(n_conv_layers):
        # create alternating convolutional and pooling layers
        if i == 0:
            model = Conv2D(
                filters=conv_nfilters,
                kernel_size=(conv_kernel_size, conv_kernel_size),
                padding="SAME",
                strides=n_strides,
                activation=activation_conv,
                input_shape=input_shape,
            )(inputs)

        else:
            model = Conv2D(
                filters=conv_nfilters,
                kernel_size=(conv_kernel_size, conv_kernel_size),
                padding="SAME",
                strides=n_strides,
                activation=activation_conv,
                input_shape=input_shape,
            )(model)

        model = MaxPool2D(pool_size=(pool_size, pool_size))(model)
    # creating dropout layer with a dropout rate corresponding to dropout_rate1
    model = Dropout(rate=dropout_rate1)(model)
    # flattening network for input into dense layer
    model = Flatten()(model)
    # doing batch normalization prior to dense layer
    model = BatchNormalization(momentum=bn_momentum)(model)
    # defining dense layer with a count defined by dense_layer1_count
    model = Dense(
        dense_layer1_count, activation=activation_dense, name="first_dense_layer",
    )(model)
    # creating dropout layer with a dropout rate corresponding to dropout_rate2
    model = Dropout(rate=dropout_rate2)(model)
    # creating another batch normalization layer
    model = BatchNormalization(momentum=bn_momentum)(model)
    # creating a final dense layer before output layers
    dense = Dense(
        dense_layer2_count, activation=activation_dense, name="second_dense_layer",
    )(model)
    # creating separate output layers for the grapheme roots (168 nodes), vowel diacritics (11 nodes) and consonant diacritics (7 nodes)
    out_root = Dense(168, activation=activation_out, name="output_root")(dense)
    out_vowel = Dense(11, activation=activation_out, name="output_vowel")(dense)
    out_consonant = Dense(7, activation=activation_out, name="output_consonant")(dense)

    # putting our model together ...
    model = Model(inputs=inputs, outputs=[out_root, out_vowel, out_consonant])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    es_root = EarlyStopping(
        monitor="output_root_accuracy", min_delta=1, patience=5, verbose=0, mode="auto"
    )
    es_vowel = EarlyStopping(
        monitor="output_vowel_accuracy", min_delta=1, patience=5, verbose=0, mode="auto"
    )
    es_consonant = EarlyStopping(
        monitor="output_consonant_accuracy",
        min_delta=1,
        patience=5,
        verbose=0,
        mode="auto",
    )

    callbacks_list = [es_root, es_vowel, es_consonant]
    # returning the model
    return model, callbacks_list
