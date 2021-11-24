"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import os
import statistics
import warnings

import trimesh
import vedo

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from vedo import load
from tqdm import tqdm

from it_clearance.testing.tester import TesterClearance
from util.util_mesh import read_sdf, find_files_mesh_env
import pandas as pd
import numpy as np
import torch.nn.functional as F
from it import util

import gc

from tabulate import tabulate

def get_next_sampling_id(l_column_names):
    return len([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")

    stratified_sampling = True

    # n_sample_per_interaction_type=382 # confidence level = 95%, margin error = 5%  for infinite samples
    n_sample_per_interaction_type=1297 # confidence level = 97%, margin error = 3%  for infinite samples
    # n_sample_per_interaction_type=2 #

    filter_dataset = "prox"    # None   prox   mp3d  replica_v1

    id = "icpst" # None   # if None it choose the next numerical id


    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filles_to_test={
        "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv")
    }


    column_prefix = f"ablation_{filter_dataset}_"
    for model in filles_to_test:

        conglo_path =filles_to_test[model]
        print(conglo_path)

        conglo_data = pd.read_csv(conglo_path)

        if id is None:
            n_sampling = get_next_sampling_id(conglo_data.columns.to_list())
        else:
            n_sampling = id

        follow_up_column = f"{column_prefix}{n_sampling}"
        conglo_data[follow_up_column] = False

        grouped = conglo_data.groupby(conglo_data['interaction_type'])


        for current_interaction_type in conglo_data['interaction_type'].unique():

            interaction_type_results = grouped.get_group(current_interaction_type)
            if stratified_sampling:
                sample = interaction_type_results.groupby('interaction_type', group_keys=False).apply(lambda x: x.sample(int(np.rint(n_sample_per_interaction_type * len(x) / len(interaction_type_results))))).sample(frac=1)
            else:
                sample = interaction_type_results.sample(n_sample_per_interaction_type)

            conglo_data.loc[sample.index.to_list(), [follow_up_column]] = True
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "non_collision"]] = ""
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "contact_sample"]] = ""
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "collision_points"]] = ""
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "collision_sum_depths"]] = ""


        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.INFO, format='')

        logging.info(f"File: {os.path.basename(os.path.realpath(__file__))}")
        logging.info(f"stratified_sampling: {stratified_sampling}")
        logging.info(f"n_sample_per_scene:  {n_sample_per_interaction_type}")
        logging.info(f"filter_dataset: {filter_dataset}")

        conglo_data.to_csv(conglo_path,index=False)
