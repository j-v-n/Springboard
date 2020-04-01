import os
import re
from collections import defaultdict
import pandas as pd


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
        # let's isolate the experiment number using regular expressions
        # exp_pattern = re.compile(r"{}/(\d\d?\d?)/.+/metrics".format(filepath))
        exp_pattern = re.compile(r"{}/([0-9]+)/.+/metrics".format(filepath))
        matches = exp_pattern.finditer(root)
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
