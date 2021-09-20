import os
import time
import pandas as pd
from os.path import  join as opj
import matplotlib.pyplot as plt

from util.util_amt_analysis import narrowing_batch_results_evaluations_test

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test/amt/batch_0001/"
    base_evaluation_file_name = "amt_unary_results"

    #############################
    # finding the last results files
    last_results_num = -1
    batch_id = ""
    for f in os.listdir(base_dir):
        if base_evaluation_file_name in f:
            s = f.replace(base_evaluation_file_name, "")
            batch_id = s[:s.find("_")]
            s = s[s.find("_")+1:]
            s = s[:s.find(".csv")]
            if s.isdigit():
                last_results_num =  max(last_results_num, int(s))

    str_last_num = str(last_results_num) if last_results_num > 0 else ""
    evaluation_file_name = batch_id+"_"+base_evaluation_file_name+str_last_num+".csv"
    evaluation_file = opj(base_dir, evaluation_file_name)
    print("Narrowing: ", evaluation_file_name)
    print("           ", evaluation_file)

    #############################
    # Reading last file of results and narrowing its data
    evaluation_data = pd.read_csv(evaluation_file)
    df_narrowed = narrowing_batch_results_evaluations_test(evaluation_data)

    #############################
    # Saving a narrow version of this data
    time_stamp_str = str(int(time.time()))
    narrowed_evaluation_file_name = batch_id + "_" + base_evaluation_file_name + str_last_num + f"_narrowed{time_stamp_str}.csv"
    narrowed_evaluation_file = opj(base_dir, narrowed_evaluation_file_name)
    df_narrowed.to_csv(opj(base_dir, narrowed_evaluation_file))