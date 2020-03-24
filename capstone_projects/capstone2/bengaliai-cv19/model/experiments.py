import mlflow
import mlflow.keras
from data_loader import data_loader
from keras.preprocessing.image import ImageDataGenerator
from batch_generator import MultiOutputDataGenerator


def run_experiment(name, model, batch_size, epochs, df_dict, data_augmentation=False):
    mlflow.keras.autolog()
    mlflow.create_experiment(name)
    mlflow.set_experiment(name)

    with mlflow.start_run():
        for i in range(4):
            print("Reading parquet file #{}".format(i + 1))
            print("-------------------------------------")
            print("Transforming data for parquet file #{}".format(i + 1))
            print("-------------------------------------")
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
                compress=True,
            )
            print("Transformation done")
            print("-------------------------------------")
            steps = x_train.shape[0] // batch_size
            if data_augmentation:

                datagen = MultiOutputDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    zca_whitening=True,
                    rotation_range=30,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                )
                print("Augmenting Input Data")
                print("--------------------")
                datagen.fit(x_train)
                print("Training model on parquet file #{}".format(i + 1))
                print("-------------------------------------")
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
                )
            print("Training finished on parquet file #{}".format(i + 1))
            print("-------------------------------------")
            print("Deleting variables after training")
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
