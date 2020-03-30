import mlflow
import mlflow.keras
from data_loader import data_loader
from keras.preprocessing.image import ImageDataGenerator
from batch_generator import MultiOutputDataGenerator


def run_experiment(
    name, model, callbacks_list, batch_size, epochs, df_dict, data_augmentation=False
):
    """
    This function is the control tower from which we run our experiments. 
    
    It loads the train and test splits from the data_loader function and trains the given model
    on the data. It also automatically logs the models, results and artifacts using mlflow and 
    saves it under the name given. 

    This enables the experiments to be visualized later using mlflow ui
    
    """
    # starting the mlflow logging
    mlflow.keras.autolog()
    mlflow.create_experiment(name)
    mlflow.set_experiment(name)

    with mlflow.start_run():
        # start a loop which goes through each of the parquet files 1 by 1
        for i in range(4):
            print("Reading parquet file #{}".format(i + 1))
            print("-------------------------------------")
            print("Transforming data for parquet file #{}".format(i + 1))
            print("-------------------------------------")
            # load the training and test splits
            (
                x_train,
                x_test,
                y_train_root,
                y_test_root,
                y_train_consonant,
                y_test_consonant,
                y_train_vowel,
                y_test_vowel,
            ) = data_loader(
                "./data/train_image_data_{}.parquet".format(i),
                df_dict["train"],
                normalize=True,
                size=(118, 68),
            )

            x_train = x_train.reshape(-1, 68, 118, 1)
            x_test = x_test.reshape(-1, 68, 118, 1)

            print("Transformation done")
            print("-------------------------------------")
            # calculate number of steps that will be used in training
            steps = x_train.shape[0] // batch_size
            if data_augmentation:
                # we will be using just the rotation and pixel shift augmentations
                datagen = MultiOutputDataGenerator(
                    rotation_range=10, width_shift_range=0.2, height_shift_range=0.2
                )
                print("Augmenting Input Data")
                print("-------------------------------------")
                # fitting the data generator to our training data
                datagen.fit(x_train)
                print("Training model on parquet file #{}".format(i + 1))
                print("-------------------------------------")
                # training the model with the generator in place
                model.fit_generator(
                    datagen.flow(
                        x_train,
                        {
                            "output_root": y_train_root,
                            "output_vowel": y_train_vowel,
                            "output_consonant": y_train_consonant,
                        },
                        batch_size=batch_size,
                    ),
                    epochs=epochs,
                    validation_data=(
                        x_test,
                        [y_test_root, y_test_vowel, y_test_consonant],
                    ),
                    steps_per_epoch=steps,
                )
            else:
                # if we don't need augmentation, we straight up train the model
                print("Training model on parquet file #{}".format(i + 1))
                print("-------------------------------------")
                model.fit(
                    x=x_train,
                    y={
                        "output_root": y_train_root,
                        "output_vowel": y_train_vowel,
                        "output_consonant": y_train_consonant,
                    },
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(
                        x_test,
                        {
                            "output_root": y_test_root,
                            "output_vowel": y_test_vowel,
                            "output_consonant": y_test_consonant,
                        },
                    ),
                    callbacks=callbacks_list,
                )
            print("Training finished on parquet file #{}".format(i + 1))
            print("-------------------------------------")
            print("Deleting variables after training")

            # finally, as per usual, delete unnecessary variables before moving to file
            del (
                x_train,
                x_test,
                y_train_root,
                y_test_root,
                y_train_consonant,
                y_test_consonant,
                y_train_vowel,
                y_test_vowel,
            )
