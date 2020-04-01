import pandas as pd
import numpy as np
from image_processor import image_processor_func


def test_func(model, name, normalize=True):
    preds_dict = {"grapheme_root": [], "vowel_diacritic": [], "consonant_diacritic": []}

    components = ["consonant_diacritic", "grapheme_root", "vowel_diacritic"]
    target = []  # model predictions placeholder
    row_id = []  # row_id place holder
    for i in range(4):
        df_test_img = pd.read_parquet("./data/test_image_data_{}.parquet".format(i))
        df_test_img.set_index("image_id", inplace=True)

        X_test = df_test_img.values.reshape(-1, 137, 236, 1)
        X_test_resized = image_processor_func(X_test, resize=True, size=(118, 68))
        X_test_resized = X_test_resized.values

        if normalize:
            # instead of loading all the data into memory and normalizing (faster but requires memory)
            # we go through the data one at a time. this is slower but does not crash your computer
            # by using up all your memory :)
            for i in range(X_test_resized.shape[0]):
                X_test_resized[i] = X_test_resized[i] / 255

        X_test_resized = X_test_resized.reshape(-1, 118, 68, 1)
        X_test_resized = X_test_resized.reshape(-1, 68, 118, 1)
        preds = model.predict(X_test_resized)

        for i, p in enumerate(preds_dict):
            preds_dict[p] = np.argmax(preds[i], axis=1)

        for k, id in enumerate(df_test_img.index.values):
            for i, comp in enumerate(components):
                id_sample = id + "_" + comp
                row_id.append(id_sample)
                target.append(preds_dict[comp][k])
        del df_test_img
        del X_test

    df_sample = pd.DataFrame(
        {"row_id": row_id, "target": target}, columns=["row_id", "target"]
    )
    df_sample.to_csv("submission_{}.csv".format(name), index=False)
