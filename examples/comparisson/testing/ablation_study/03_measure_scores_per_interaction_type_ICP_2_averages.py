"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import os
import statistics
import warnings

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from tqdm import tqdm

import pandas as pd

from tabulate import tabulate


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")


    filter_dataset = "prox"    # None   prox   mp3d  replica_v1


    id = "icpst"  # None   # if None it choose the next numerical id



    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filles_to_test = {
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv")
    }


    column_prefix = f"ablation_{filter_dataset}_"
    for model in filles_to_test:
        tb_headers = ["model", "dataset", "interaction_type", "non_collision","std_dev", "contact", "collision_points","collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
        tb_data = []

        conglo_path =filles_to_test[model]
        print(conglo_path)

        loss_non_collisions_model=[]
        loss_contacts_model=[]
        loss_collision_n_points_model = []
        loss_collision_sum_depths_model = []

        conglo_data = pd.read_csv(conglo_path)


        follow_up_column = f"{column_prefix}{id}"

        grouped = conglo_data.groupby(conglo_data['interaction_type'])


        for current_interaction_type in conglo_data['interaction_type'].unique():

            loss_non_collision_inter_type, loss_contact_inter_type = [], []
            loss_collision_n_points, loss_collision_sum_depths = [], []

            interaction_type_results = grouped.get_group(current_interaction_type)

            sample = interaction_type_results[interaction_type_results[follow_up_column] == True]


            for idx, row in tqdm(sample.iterrows(), total=sample.shape[0]):

                current_loss_non_coll = sample.loc[idx, [follow_up_column + "non_collision"]].values[0]
                current_loss_contact = sample.loc[idx, [follow_up_column + "contact_sample"]].values[0]
                current_contact_n_points = sample.loc[idx, [follow_up_column + "collision_points"]].values[0]
                current_contact_sum_depths = sample.loc[idx, [follow_up_column + "collision_sum_depths"]].values[0]

                loss_non_collision_inter_type.append(current_loss_non_coll)
                loss_contact_inter_type.append(current_loss_contact)
                loss_collision_n_points.append(current_contact_n_points)
                loss_collision_sum_depths.append(current_contact_sum_depths)

                loss_non_collisions_model.append(current_loss_non_coll)
                loss_contacts_model.append(current_loss_contact)
                loss_collision_n_points_model.append(current_contact_n_points)
                loss_collision_sum_depths_model.append(current_contact_sum_depths)

            tb_data.append([model, filter_dataset, current_interaction_type,
                            statistics.mean(loss_non_collision_inter_type),
                            statistics.stdev(loss_non_collision_inter_type),
                            statistics.mean(loss_contact_inter_type),
                            statistics.mean(loss_collision_n_points),
                            statistics.stdev(loss_collision_n_points),
                            statistics.mean(loss_collision_sum_depths),
                            statistics.stdev(loss_collision_sum_depths)])

        tb_data.append([model, filter_dataset, "Overall",
                        statistics.mean(loss_non_collisions_model),
                        statistics.stdev(loss_non_collisions_model),
                        statistics.mean(loss_contacts_model),
                        statistics.mean(loss_collision_n_points_model),
                        statistics.stdev(loss_collision_n_points_model),
                        statistics.mean(loss_collision_sum_depths_model),
                        statistics.stdev(loss_collision_sum_depths_model)])


        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.INFO, format='')

        logging.info(f"File: {os.path.basename(os.path.realpath(__file__))}")
        logging.info(f"filter_dataset: {filter_dataset}")

        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))


