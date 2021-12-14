"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import argparse
import os
import statistics
import warnings

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch

import pandas as pd

from tabulate import tabulate


parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--follow_up_file_name', required=True, help='file name with the follow up')
opt = parser.parse_args()
print(opt)


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")

    stratified_sampling = True

    id = "itClearance"  # None

    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")


    column_prefix = "piloting_"
    model = "conglo_env_fill_iT_clearance"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_headers = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact",
                  "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
    tb_headers_optim = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact",
                  "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
    tb_data = []
    tb_data_optim = []

    conglo_path = opj(env_filled_data_test_dir, "piloting", opt.follow_up_file_name)
    print(conglo_path)

    conglo_data = pd.read_csv(conglo_path)

    follow_up_column = f"{column_prefix}{id}"

    for dataset_name in conglo_data['dataset'].unique():

        per_dataset_data = conglo_data[conglo_data['dataset'] == dataset_name]

        loss_non_collisions_dataset = []
        loss_contacts_dataset = []
        loss_collision_n_points_dataset = []
        loss_collision_sum_depths_dataset = []

        loss_non_collisions_dataset_optim = []
        loss_contacts_dataset_optim = []
        loss_collision_n_points_dataset_optim = []
        loss_collision_sum_depths_dataset_optim = []


        grouped = per_dataset_data.groupby('scene')

        for current_env_name in per_dataset_data['scene'].unique():

            loss_non_collision_env, loss_contact_env = [], []
            loss_collision_n_points, loss_collision_sum_depths = [], []
            loss_non_collision_env_optim, loss_contact_env_optim = [], []
            loss_collision_n_points_optim, loss_collision_sum_depths_optim = [], []

            dataset_results = grouped.get_group(current_env_name)

            sample = dataset_results[dataset_results[follow_up_column] == True]

            for idx, row in sample.iterrows():

                current_loss_non_coll = sample.loc[idx, [follow_up_column + "_non_collision"]].values[0]
                current_loss_contact = sample.loc[idx, [follow_up_column + "_contact_sample"]].values[0]
                current_contact_n_points = sample.loc[idx, [follow_up_column + "_collision_points"]].values[0]
                current_contact_sum_depths = sample.loc[idx, [follow_up_column + "_collision_sum_depths"]].values[0]

                loss_non_collision_env.append( current_loss_non_coll)
                loss_contact_env.append(current_loss_contact)
                loss_collision_n_points.append(current_contact_n_points)
                loss_collision_sum_depths.append(current_contact_sum_depths)

                loss_non_collisions_dataset.append(current_loss_non_coll)
                loss_contacts_dataset.append(current_loss_contact)
                loss_collision_n_points_dataset.append(current_contact_n_points)
                loss_collision_sum_depths_dataset.append(current_contact_sum_depths)

                current_loss_non_coll_optim = sample.loc[idx, [follow_up_column + "_non_collision_optim"]].values[0]
                current_loss_contact_optim = sample.loc[idx, [follow_up_column + "_contact_sample_optim"]].values[0]
                current_contact_n_points_optim = sample.loc[idx, [follow_up_column + "_collision_points_optim"]].values[0]
                current_contact_sum_depths_optim = sample.loc[idx, [follow_up_column + "_collision_sum_depths_optim"]].values[0]

                loss_non_collision_env_optim.append(current_loss_non_coll_optim)
                loss_contact_env_optim.append(current_loss_contact_optim)
                loss_collision_n_points_optim.append(current_contact_n_points_optim)
                loss_collision_sum_depths_optim.append(current_contact_sum_depths_optim)

                loss_non_collisions_dataset_optim.append(current_loss_non_coll_optim)
                loss_contacts_dataset_optim.append(current_loss_contact_optim)
                loss_collision_n_points_dataset_optim.append(current_contact_n_points_optim)
                loss_collision_sum_depths_dataset_optim.append(current_contact_sum_depths_optim)

            tb_data.append([model, dataset_name, current_env_name,
                            statistics.mean(loss_non_collision_env),
                            statistics.stdev(loss_non_collision_env),
                            statistics.mean(loss_contact_env),
                            statistics.mean(loss_collision_n_points),
                            statistics.stdev(loss_collision_n_points),
                            statistics.mean(loss_collision_sum_depths),
                            statistics.stdev(loss_collision_sum_depths)
                            ])
            tb_data_optim.append([model, dataset_name, current_env_name,
                                  statistics.mean(loss_non_collision_env_optim),
                                  statistics.stdev(loss_non_collision_env_optim),
                                  statistics.mean(loss_contact_env_optim),
                                  statistics.mean(loss_collision_n_points_optim),
                                  statistics.stdev(loss_collision_n_points_optim),
                                  statistics.mean(loss_collision_sum_depths_optim),
                                  statistics.stdev(loss_collision_sum_depths_optim)
                                  ])

        tb_data.append([model, dataset_name, "Overall",
                        statistics.mean(loss_non_collisions_dataset),
                        statistics.stdev(loss_non_collisions_dataset),
                        statistics.mean(loss_contacts_dataset),
                        statistics.mean(loss_collision_n_points_dataset),
                        statistics.stdev(loss_collision_n_points_dataset),
                        statistics.mean(loss_collision_sum_depths_dataset),
                        statistics.stdev(loss_collision_sum_depths_dataset)])

        tb_data_optim.append([model, dataset_name, "Overall",
                              statistics.mean(loss_non_collisions_dataset_optim),
                              statistics.stdev(loss_non_collisions_dataset_optim),
                              statistics.mean(loss_contacts_dataset_optim),
                              statistics.mean(loss_collision_n_points_dataset_optim),
                              statistics.stdev(loss_collision_n_points_dataset_optim),
                              statistics.mean(loss_collision_sum_depths_dataset_optim),
                              statistics.stdev(loss_collision_sum_depths_dataset_optim)])

    import logging
    conglo_file_name = os.path.basename(conglo_path)
    logging.basicConfig(filename=f"output_{conglo_file_name}.txt", level=logging.INFO, format='')

    logging.info('\n\n\n')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n RAW iTClearance OUTPUT')
    logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
    logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))

    logging.info('\n\n\n')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n OPTIMIZED OUTPUT')
    logging.info('\n'+tabulate(tb_data_optim,headers=tb_headers_optim, floatfmt=".4f",  tablefmt="latex_booktabs"))
    logging.info('\n'+tabulate(tb_data_optim,headers=tb_headers_optim, floatfmt=".4f",  tablefmt="simple"))
