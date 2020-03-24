import pandas as pd
import numpy as np


def test_func(model, name):
    preds_dict = {"grapheme_root": [], "vowel_diacritic": [], "consonant_diacritic": []}

    components = ["consonant_diacritic", "grapheme_root", "vowel_diacritic"]
    target = []  # model predictions placeholder
    row_id = []  # row_id place holder
    for i in range(4):
        df_test_img = pd.read_parquet("./data/test_image_data_{}.parquet".format(i))
        df_test_img.set_index("image_id", inplace=True)

        X_test = df_test_img.values.reshape(-1, 137, 236, 1)

        preds = model.predict(X_test)

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
