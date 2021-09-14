import os

import pandas as pd
from os.path import  join as opj
import matplotlib.pyplot as plt



if __name__ == '__main__':


    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test/amt/results"

    comparisson_file_paths = []

    ##########
    # compiling evaluation data
    ##########
    evaluation_file_paths = []
    for csv_file in os.listdir(base_dir):
        if csv_file.startswith("comparison_"):
            comparisson_file_paths.append(opj(base_dir, csv_file))
        if csv_file.startswith("evaluation_"):
            evaluation_file_paths.append(opj(base_dir, csv_file))
    #read first file
    evaluation_data=pd.read_csv(evaluation_file_paths.pop(0))
    for csv_file_path in evaluation_file_paths:
        data = pd.read_csv(csv_file_path)
        evaluation_data = pd.concat([evaluation_data, data], ignore_index=True)
    evaluation_data.to_csv(opj(base_dir,"compilation_evaluation.csv"))
    ##########
    # frequency table for evaluation data
    ##########
    approved_interaction_data = evaluation_data[evaluation_data["AssignmentStatus"] == "Approved"]
    frequency_table = approved_interaction_data.groupby(["Input.order"])[
        ["Answer.strongly_disagree.on", "Answer.disagree.on", "Answer.neither.on", "Answer.agree.on",
         "Answer.strongly_agree.on"]].sum()
    frequency_table = frequency_table.rename(columns={"Answer.strongly_disagree.on": "strongly\ndisagree",
                                                      "Answer.disagree.on": "disagree",
                                                      "Answer.neither.on": "neither\ndisagree\nnor agree",
                                                      "Answer.agree.on": "agree",
                                                      "Answer.strongly_agree.on": "strongly\nagree"})
    # Total sum per column:
    frequency_table.loc['Total', :] = frequency_table.sum(axis=0)
    # Total sum per row:
    frequency_table.loc[:, 'Total'] = frequency_table.sum(axis=1)
    evaluation_data.to_csv(opj(base_dir, "compilation_evaluation_frequencies.csv"))

    ################# save here

    ##########
    # ploting frequencies of evaluation data
    ##########
    pd_for_plot = frequency_table.T
    del pd_for_plot['fake']
    del pd_for_plot['Total']
    pd_for_plot = pd_for_plot.drop(index=['Total'])
    pd_for_plot.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.margins(0.2, tight=False)
    plt.show()

