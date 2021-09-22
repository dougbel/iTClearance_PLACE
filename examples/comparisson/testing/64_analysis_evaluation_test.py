import os
import time
import pandas as pd
from os.path import  join as opj
import matplotlib.pyplot as plt

from util.util_amt_analysis import narrowing_batch_results_evaluations_test

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test/amt/results/"
    base_evaluation_file_name = "amt_unary_final_results"

    file_names_to_compile = []
    min_batch = 10000
    max_batch = 0
    for f in os.listdir(base_dir):
        if base_evaluation_file_name in f:
            file_names_to_compile.append(f)
            batch_id = int(f[:f.find("_")])
            min_batch =  min(min_batch, batch_id)
            max_batch = max(max_batch, batch_id)


    l_approved_to_join =[]
    for name in file_names_to_compile:
        evaluation_file = opj(base_dir, name)
        evaluation_data = pd.read_csv(evaluation_file)
        df_narrowed = narrowing_batch_results_evaluations_test(evaluation_data)
        df_approved_data = df_narrowed[df_narrowed["AssignmentStatus"]=="Approved"]
        l_approved_to_join.append(df_approved_data)
        print("Narrowed: ", name)
        print("          ", evaluation_file)

    df_approved_results = pd.concat(l_approved_to_join, ignore_index=True)


    # ##########
    # # frequency table for evaluation data
    frequency_table = df_approved_results.groupby(["order"])[ ["strongly_disagree", "disagree", "neither", "agree", "strongly_agree"]].sum()
    frequency_table.reset_index(inplace=True)
    frequency_table.loc[frequency_table["order"] == "it", "order"] = "iTClearance"
    frequency_table.loc[frequency_table["order"] == "place", "order"] = "PLACE"
    frequency_table.set_index("order", inplace=True)
    frequency_table.index.name = None
    frequency_table = frequency_table.rename(columns={"strongly_disagree": "strongly\ndisagree", "strongly_agree": "strongly\nagree"})
    # Total sum per column:
    frequency_table.loc['Total', :] = frequency_table.sum(axis=0)
    # Total sum per row:
    frequency_table.loc[:, 'Total'] = frequency_table.sum(axis=1)


    # ##########
    # # ploting frequencies of evaluation data
    pd_for_plot = frequency_table.T
    del pd_for_plot['fake']
    del pd_for_plot['Total']
    pd_for_plot = pd_for_plot.drop(index=['Total'])
    pd_for_plot.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.margins(0.2, tight=False)


    output_base_name = f"{min_batch}-{max_batch}_amt_unary_final_results"
    output_dir = opj(base_dir,"compilations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_approved_results.to_csv(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed.csv"))
    frequency_table.to_csv(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed_frequencies.csv"))
    plt.savefig(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed_frequencies.png"))
