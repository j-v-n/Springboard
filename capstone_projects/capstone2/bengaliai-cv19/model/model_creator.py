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


def model_create(
    input_shape=(137, 236, 1),
    n_conv_layers=3,
    dropout_rate1=0.3,
    dense_layer1_count=1024,
    dropout_rate2=0.3,
    dense_layer2_count=512,
    conv_nfilters=64,
    conv_kernel_size=3,
    n_strides=0,
    pool_size=5,
    activation_conv="relu",
    activation_dense="relu",
    activation_out="softmax",
):

    """
    Function to create a CNN with alternating convolutional and pooling layers,
    followed by a dropout, then 2 dense layers with a dropout in between and finally 3 output layers for the 3 target variables,
    i.e grapheme root, vowel diacritic and consonant diacritic
    
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
                activation=activation_conv,
                input_shape=input_shape,
            )(inputs)

        else:
            model = Conv2D(
                filters=conv_nfilters,
                kernel_size=(conv_kernel_size, conv_kernel_size),
                padding="SAME",
                activation=activation_conv,
                input_shape=input_shape,
            )(model)

        model = MaxPool2D(pool_size=(pool_size, pool_size), dim_ordering="tf")(model)

    model = Dropout(rate=dropout_rate1)(model)
    model = Flatten()(model)
    model = Dense(
        dense_layer1_count, activation=activation_dense, name="first_dense_layer"
    )(model)
    model = Dropout(rate=dropout_rate2)(model)
    dense = Dense(
        dense_layer2_count, activation=activation_dense, name="second_dense_layer"
    )(model)

    out_root = Dense(168, activation=activation_out, name="output_root")(dense)
    out_vowel = Dense(11, activation=activation_out, name="output_vowel")(dense)
    out_consonant = Dense(7, activation=activation_out, name="output_consonant")(dense)

    model = Model(inputs=inputs, outputs=[out_root, out_vowel, out_consonant])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
