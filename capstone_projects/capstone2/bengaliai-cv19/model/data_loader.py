import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from image_processor import image_processor_func


def data_loader(parquet_file_path, df_dict_file, normalize=True, size=(118, 68)):
    """
    This function loads up each parquet file, merges with corresponding target values
    , compresses the feature space from 137*236 to a new resolution (by default it goes to
    half the original resolution, i.e 68*118) and normalizes the data if the user chooses to
    
    """
    # read parquet file and then merge it with the dataframe
    train_images_df = pd.read_parquet(parquet_file_path)

    # merge the parquet file data which contains the actual features with the target variables
    train_df = pd.merge(train_images_df, df_dict_file, on="image_id").drop(
        ["image_id"], axis=1
    )

    # delete the parquet file dataframe to save memory
    del train_images_df

    # extract X values from the newly created dataframe - this will have 137*236 columns
    X = train_df.drop(
        ["grapheme_root", "vowel_diacritic", "consonant_diacritic", "grapheme"], axis=1
    ).values

    # extract y values - these will only have 1 column each
    y_root = train_df.grapheme_root.values
    y_vowel = train_df.vowel_diacritic.values
    y_consonant = train_df.consonant_diacritic.values

    # delete the dataframe since we no longer need it
    del train_df

    # since we have multiple choices for each output, we must OneHotEncode the target variables
    # i.e categorize them. Remember to use sparse=False. Otherwise the model will not run
    ohe = OneHotEncoder(sparse=False)
    y_root = ohe.fit_transform(y_root.reshape(-1, 1))
    y_vowel = ohe.fit_transform(y_vowel.reshape(-1, 1))
    y_consonant = ohe.fit_transform(y_consonant.reshape(-1, 1))

    # create train and test sets from yhis data
    (
        X_train,
        X_test,
        y_train_root,
        y_test_root,
        y_train_consonant,
        y_test_consonant,
        y_train_vowel,
        y_test_vowel,
    ) = train_test_split(X, y_root, y_consonant, y_vowel, test_size=0.1)

    # using the image_processor_func defined in the custom image processor module (built using OpenCV),
    # we apply thresholding filters to the image and then compress the images to the size specified
    print("Compressing Images")
    print("-------------------------------------")
    X_train = X_train.reshape(-1, 137, 236, 1)
    X_test = X_test.reshape(-1, 137, 236, 1)
    X_train_resized = image_processor_func(X_train, resize=True, size=size)  # dataframe
    X_test_resized = image_processor_func(X_test, resize=True, size=size)  # dataframe

    # extracting only values from the dataframe - this makes normalization much faster
    X_train_resized = X_train_resized.values
    X_test_resized = X_test_resized.values

    # Normalizing images - if needed. Highly recommended
    if normalize:
        # instead of loading all the data into memory and normalizing (faster but requires memory)
        # we go through the data one at a time. this is slower but does not crash your computer
        # by using up all your memory :)
        for i in range(X_train_resized.shape[0]):
            X_train_resized[i] = X_train_resized[i] / 255

        for i in range(X_test_resized.shape[0]):
            X_test_resized[i] = X_test_resized[i] / 255

    # we now reshape this data to the corresponding dimensions based on the new size of the images
    X_train_resized = X_train_resized.reshape(-1, size[0], size[1], 1)
    X_test_resized = X_test_resized.reshape(-1, size[0], size[1], 1)

    # finally let's delete some large matrices to save memory
    del y_root
    del y_consonant
    del y_vowel
    del X
    del X_train
    del X_test

    # aaaaand we finally return our train, test features and targets! phew!
    return (
        X_train_resized,
        X_test_resized,
        y_train_root,
        y_test_root,
        y_train_consonant,
        y_test_consonant,
        y_train_vowel,
        y_test_vowel,
    )
