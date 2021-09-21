import os
import time
import pandas as pd
from os.path import  join as opj
import matplotlib.pyplot as plt

from util.util_amt_analysis import narrowing_batch_results_comparison_test

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test/amt/results/"
    base_evaluation_file_name = "amt_binary_final_results"

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
        df_narrowed = narrowing_batch_results_comparison_test(evaluation_data)
        df_approved_data = df_narrowed[df_narrowed["AssignmentStatus"]=="Approved"]
        l_approved_to_join.append(df_approved_data)
        print("Narrowed: ", name)
        print("          ", evaluation_file)

    df_approved_results = pd.concat(l_approved_to_join, ignore_index=True)

    df_approved_results["it_voted"] = 0
    df_approved_results["place_voted"] = 0

    for index, row in df_approved_results.iterrows():
        if row["order"] == "it_place":
            if row["exampleA"] ==True:
                df_approved_results.at[index, "it_voted"] = 1
            else:
                df_approved_results.at[index, "place_voted"] = 1

        if row["order"] == "place_it":
            if row["exampleA"] == True:
                df_approved_results.at[index, "place_voted"] = 1
            else:
                df_approved_results.at[index, "it_voted"] = 1



    # ##########
    # # frequency table for evaluation data
    frequency_table = df_approved_results.groupby(["dataset"])[ ["it_voted", "place_voted"]].sum()

    frequency_table = frequency_table.rename(columns={"it_voted": "iTClearance", "place_voted": "PLACE"})

    frequency_table=frequency_table.T
    # Total sum per column:
    frequency_table.loc['Total', :] = frequency_table.sum(axis=0)
    # Total sum per row:
    frequency_table.loc[:, 'Total'] = frequency_table.sum(axis=1)


    # ##########
    # # ploting frequencies of evaluation data
    pd_for_plot = frequency_table.T
    del pd_for_plot['Total']
    pd_for_plot = pd_for_plot.drop(index=['Total'])
    pd_for_plot.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.margins(0.2, tight=False)


    output_base_name = f"{min_batch}-{max_batch}_amt_binary_final_results"
    output_dir = opj(base_dir,"compilations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_approved_results.to_csv(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed.csv"))
    frequency_table.to_csv(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed_frequencies.csv"))
    plt.savefig(opj(output_dir, f"{output_base_name}_onlyapproved_and_narrowed_frequencies.png"))
