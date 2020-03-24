import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from collections import defaultdict
import mlflow
import random

random.seed(42)
# using custom scripts for creating model and loading data
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "./model/")

from model_creator import model_create
from data_loader import data_loader

from experiments import run_experiment
from tester import test_func

from batch_generator import MultiOutputDataGenerator

from keras import backend as K

K.tensorflow_backend._get_available_gpus()


# read the csv files
filenames = [
    "train",
    "test",
    "class_map",
    "class_map_corrected",
    "train_multi_diacritics",
    "sample_submission",
]
df_dict = defaultdict()

for file in filenames:
    df_dict[file] = pd.read_csv("../data/{}.csv".format(file))


import mlflow.keras

print("Creating Model")
model = model_create(
    input_shape=(64, 37, 1),
    conv_kernel_size=4,
    pool_size=2,
    dropout_rate1=0.5,
    dropout_rate2=0.5,
    n_conv_layers=3,
)
run_experiment(
    "compress_exp_aug_yes_conv_layers_3_conv_kernel_size_4_pool_size_2_dropout_rate_0.5",
    model,
    batch_size=100,
    epochs=50,
    df_dict=df_dict,
    data_augmentation=True,
)
