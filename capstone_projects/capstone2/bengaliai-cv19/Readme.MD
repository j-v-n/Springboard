# Capstone Project 2 - Bengali Grapheme Classification Project

By Jayanth Nair

The aim of this capstone project is to develop a Deep Neural Network Classifier for classifying Handwritten Bengali Characters. 

# Data
The dataset used for this project is from this [Kaggle Competition](https://www.kaggle.com/c/bengaliai-cv19) run by Bengali.AI

# Notebooks and Scripts
In this project, the notebooks are only used to visualize the data and interactively run the experiments. Most of the functional code has been modularized as Python scripts stored in the models folder. The scripts, notebooks and brief descriptions are provided below
  - Scripts
    - data_loader -> Loads data in each parquet file provided as part of the data set, one hot encodes labels, compresses input images into half the original resolution for faster training and normalizes the data
    - experiments -> Sets up experiments using the MLflow API for tracking (augmentation is optional)
    - image_processor -> Using the OpenCV library, this script resizes the images (called in data_loader) and applies threshold filtering
    - model_creator -> Creates a ConvNet architecture which takes number of convolutional layers, kernel size, dropout rates, dense layer densities and input shape.  Uses batch normalization by default. The architecture involves alternating convolutional and pooling layers
    - run_checker -> Navigates the mlruns directory created by MLflow, reads all the test results using RegEx and populates a dataframe with the final experiment results
    - tester -> Used for inference on test data and creating output submission csv 
    - debugger -> Used on an as-needed basis for debugging

  - Notebooks
    - visualization -> Exploratory notebook which visualizes features within the dataset
    - experiments -> First set of  experiments with a 108 experiment-long matrix. This set of experiments does not consider image augmentation
    - experiments_second_set_augmentation -> Second set of experiments with the top 25 experiments from the first set, with image augmentation

# Reports
  - Milestone Report -> A summary of the EDA and data wrangling process
  - Final Report -> Final report on the project

# Acknowledgements
Just like for Capstone 1, apart from coffee and early mornings, this project wouldn't have been possible without:
- [Ankur Verma](https://github.com/ankurv857), my Springboard Mentor
- [This Starter Kaggle Notebook](https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn), for the ideas on multioutput generators amongst other things
- [Leslie Frederick](https://github.com/lfrederick2106), for her infinite patience