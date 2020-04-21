import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from run_checker import metrics_trends


def plot_and_save(model_list):
    plt.style.use("seaborn-bright")
    plot_dict = {0: "output_", 1: "val_output_"}
    title_dict = {0: "training_set_", 1: "validation_set_"}
    label_dict = {0: "accuracy", 1: "loss"}
    for model in model_list:
        trends = metrics_trends(model)
        os.chdir(
            "/home/jayanth/Documents/springboard/capstone_projects/capstone2/bengaliai-cv19/"
        )

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 6))

        for idx, ax in enumerate(axes):
            for plot_num in range(2):
                ax[plot_num].plot(
                    trends["cumtime"],
                    trends[plot_dict[idx] + "vowel_" + label_dict[plot_num]],
                    label="vowel",
                )
                ax[plot_num].plot(
                    trends["cumtime"],
                    trends[plot_dict[idx] + "root_" + label_dict[plot_num]],
                    label="root",
                )
                ax[plot_num].plot(
                    trends["cumtime"],
                    trends[plot_dict[idx] + "consonant_" + label_dict[plot_num]],
                    label="consonant",
                )
                ax[plot_num].set_ylabel(label_dict[plot_num] + " (-)")
                ax[plot_num].set_title(title_dict[idx] + label_dict[plot_num])
                ax[plot_num].legend(loc="center right")
                if idx == 0:
                    ax[plot_num].tick_params(
                        axis="x", which="both", bottom=False, top=False
                    )

        fig.suptitle("Training Trends for Model #{}".format(model))

        plt.savefig("model_{}_eval.png".format(model))


model_list = [57, 33, 56, 60, 63, 55]

plot_and_save(model_list)
