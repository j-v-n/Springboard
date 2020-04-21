import os
import re
from collections import defaultdict
import pandas as pd
import yaml


def results_accumulator(
    filepath="/home/jayanth/Documents/springboard/capstone_projects/capstone2/bengaliai-cv19/mlruns",
):
    """
    This helper function navigates all the folders from the different mlflow experiments and populates
    a dictionary with the final training and validation accuracies in a dictionary, which is converted
    and returned as a dataframe

    Arguments:
        filepath - Location of mlruns folder

    Returns:
        master_exp_df - A pandas dataframe with experiment numbers and accuracies

    """

    # first let's change the current working directory
    os.chdir(filepath)

    # let's initiate an empty dictionary
    master_exp_dict = defaultdict(dict)

    # let's navigate through each of the subfolders and files within the mlruns folder
    for root, _, files in os.walk(os.getcwd()):

        for f in files:
            # the experiment number must be read from the meta yaml file using regular expressions
            if f == "meta.yaml":
                os.chdir(root)
                yaml_dict = yaml.load(open(f))
                if yaml_dict["name"] != "":
                    exp_pattern = re.compile(r"exp_(\d\d?)_.")
                    matches = exp_pattern.finditer(yaml_dict["name"])
                    for match in matches:
                        exp_number = match.group(1)

        # navigating into the metrics folder for each experiment
        if root.endswith("metrics"):
            os.chdir(root)
            # moving into that directory
            for f in files:
                # iterating through list of files and opening only the ones which list accuracy
                if f.endswith("accuracy"):
                    fname = f
                    # need to open in binary mode as file is not .txt
                    with open(fname, "rb") as f_file:
                        # this bit of code comes from here : https://stackoverflow.com/questions/46258499/read-the-last-line-of-a-file-in-python
                        # it seeks out the end of the file and reads last line
                        f_file.seek(-2, os.SEEK_END)
                        while f_file.read(1) != b"\n":
                            f_file.seek(-2, os.SEEK_CUR)
                        last_line = f_file.readline().decode()
                        acc_pattern = re.compile(r"0\.\d+")
                        acc_matches = acc_pattern.finditer(last_line)
                        # appending to our dictionary
                        for match in acc_matches:
                            master_exp_dict[exp_number][f] = match[0]
    # converting dictionary to a dataframe
    master_exp_df = pd.DataFrame.from_dict(master_exp_dict, orient="index")
    master_exp_df.index.name = "experiment_number"

    # also return back to original project path
    os.chdir(
        "/home/jayanth/Documents/springboard/capstone_projects/capstone2/bengaliai-cv19/"
    )
    # aaaaand finally returning our dataframe
    return master_exp_df


def metrics_trends(
    model_number,
    filepath="/home/jayanth/Documents/springboard/capstone_projects/capstone2/bengaliai-cv19/mlruns",
    metrics_list=[
        "output_root_accuracy",
        "output_root_loss",
        "val_output_root_accuracy",
        "output_vowel_accuracy",
        "output_vowel_loss",
        "val_output_vowel_accuracy",
        "output_consonant_accuracy",
        "output_consonant_loss",
        "val_output_consonant_accuracy",
        "val_output_root_loss",
        "val_output_vowel_loss",
        "val_output_consonant_loss",
    ],
):
    """
    This helper function navigates all the folders from the different mlflow experiments and populates
    a dictionary with the trends of different metrics for each model number

    Arguments:
        filepath - Location of mlruns folder
        model_number - Model number
        metrics_list - List of metrics

    """

    # first let's change the current working directory
    os.chdir(filepath)

    # let's initiate an empty dictionary
    trends_dict = defaultdict(dict)

    time_flag = 0
    time_list = []
    # let's navigate through each of the subfolders and files within the mlruns folder
    for root, _, files in os.walk(os.getcwd()):

        for f in files:
            # the experiment number must be read from the meta yaml file using regular expressions
            if f == "meta.yaml":
                os.chdir(root)
                yaml_dict = yaml.load(open(f), Loader=yaml.FullLoader)
                if yaml_dict["name"] != "":
                    exp_pattern = re.compile(r"exp_(\d\d?)_.")
                    matches = exp_pattern.finditer(yaml_dict["name"])
                    for match in matches:
                        exp_number = match.group(1)
                        if int(exp_number) == model_number:
                            name_pattern = re.compile(r"/mlruns/(\d\d?)")
                            matches = name_pattern.finditer(root)
                            for match in matches:
                                folder_number = match.group(1)

        try:
            if "mlruns/{}/".format(folder_number) in root:
                if root.endswith("metrics"):
                    os.chdir(root)
                    for f in files:
                        if f in metrics_list:
                            with open(f, "rb") as f_file:
                                accuracy = []

                                for line in f_file:
                                    acc_pattern = re.compile(r"\d\.\d+")

                                    line = line.decode()
                                    acc_matches = acc_pattern.finditer(line)

                                    for acc_match in acc_matches:
                                        accuracy.append(float(acc_match[0]))

                                    if time_flag == 0:
                                        time_pattern = re.compile(r"1586\d{9}")
                                        time_matches = time_pattern.finditer(line)
                                        for time_match in time_matches:
                                            time_list.append(time_match[0])

                                if time_flag == 0:
                                    trends_dict["time"] = time_list
                                    time_flag += 1
                                trends_dict[f] = accuracy

        except UnboundLocalError:
            pass

    df = pd.DataFrame.from_dict(trends_dict)
    df["time"] = df["time"].astype(int)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["cumtime"] = pd.to_timedelta(df["time"] - df["time"][0]).astype("timedelta64[s]")

    return df
