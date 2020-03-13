import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def data_loader(parquet_file_path, df_dict_file):
    """
    This function loads up each parquet file, merges with corresponding target values
    and returns X and y arrays for training by the neural network
    
    """
    # read parquet file and then merge it with the dataframe
    train_images_df = pd.read_parquet(parquet_file_path)

    train_df = pd.merge(train_images_df, df_dict_file, on="image_id").drop(
        ["image_id"], axis=1
    )
    del train_images_df
    # extract X values
    X = train_df.drop(
        ["grapheme_root", "vowel_diacritic", "consonant_diacritic", "grapheme"], axis=1
    ).values

    X = X.reshape(-1, 137, 236, 1)

    # extract y values
    y_root = train_df.grapheme_root.values
    y_vowel = train_df.vowel_diacritic.values
    y_consonant = train_df.consonant_diacritic.values

    del train_df
    # onehotencode y values
    ohe = OneHotEncoder()
    y_root = ohe.fit_transform(y_root.reshape(-1, 1))
    y_vowel = ohe.fit_transform(y_vowel.reshape(-1, 1))
    y_consonant = ohe.fit_transform(y_consonant.reshape(-1, 1))

    # create train and test sets
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

    # delete some large arrays to save memory
    del y_root
    del y_consonant
    del y_vowel
    del X

    return (
        X_train,
        X_test,
        y_train_root,
        y_test_root,
        y_train_consonant,
        y_test_consonant,
        y_train_vowel,
        y_test_vowel,
    )

