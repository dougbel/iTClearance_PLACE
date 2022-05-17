"""
Same sampling positions would be use for the PLACE execution to have a fair comparison over the same elements in
the environment
"""
import argparse
import os
import statistics
import warnings

from numpy.distutils.command.config import config
from tabulate import tabulate
import numpy as np

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import pandas as pd
import math


def get_next_sampling_id(l_column_names):
    return len([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="mp3d", help='scene')
opt = parser.parse_args()
print(opt)


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    studies_dir = opj(base_dir, "comparison_study_test", "bubble_fillers")

    output_file_name_pattern = "03_conglomerate_capable_positions_clearance_"
    model = "itClearance++"
    id = "itClearance"  # None

    column_prefix = f"comparison_"
    follow_up_column = f"{column_prefix}{id}_improved"

    tb_headers = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact","contact_std",
                  "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
    tb_headers_optim = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact","contact_std",
                        "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
    tb_data_orig = []
    tb_data_optim = []


    loss_non_collisions_dataset_orig = []
    loss_contacts_dataset_orig = []
    loss_collision_n_points_dataset_orig = []
    loss_collision_sum_depths_dataset_orig = []

    loss_non_collisions_dataset_optim = []
    loss_contacts_dataset_optim = []
    loss_collision_n_points_dataset_optim = []
    loss_collision_sum_depths_dataset_optim = []

    

    for file in os.listdir(studies_dir):

        # avoid files that are not part of studies
        if not file.startswith(output_file_name_pattern):
            continue
        conglo_data_scene = pd.read_csv(opj(studies_dir,file))

        # avoid file results that are not part of the dataset
        if opt.dataset not in conglo_data_scene['dataset'].unique():
            continue
        conglo_data_scene = conglo_data_scene[conglo_data_scene['dataset']==opt.dataset]

        current_env_name = conglo_data_scene['scene'].unique()[0]

        print(file)
        print(opt.dataset, " - ", current_env_name)

        loss_non_collision_env_orig, loss_contact_env_orig = [], []
        loss_collision_n_points_orig, loss_collision_sum_depths_orig = [], []
        loss_non_collision_env_optim, loss_contact_env_optim = [], []
        loss_collision_n_points_optim, loss_collision_sum_depths_optim = [], []


        conglo_data = conglo_data_scene[conglo_data_scene[follow_up_column] == True]

        for idx, row in conglo_data.iterrows():

            current_loss_non_coll_orig = conglo_data.loc[idx, [follow_up_column + "_non_collision"]].values[0]
            current_loss_contact_orig = conglo_data.loc[idx, [follow_up_column + "_contact_sample"]].values[0]
            current_contact_n_points_orig = conglo_data.loc[idx, [follow_up_column + "_collision_points"]].values[0]
            current_contact_sum_depths_orig = conglo_data.loc[idx, [follow_up_column + "_collision_sum_depths"]].values[0]

            loss_non_collision_env_orig.append(current_loss_non_coll_orig)
            loss_contact_env_orig.append(current_loss_contact_orig)
            loss_collision_n_points_orig.append(current_contact_n_points_orig)
            loss_collision_sum_depths_orig.append(current_contact_sum_depths_orig)

            loss_non_collisions_dataset_orig.append(current_loss_non_coll_orig)
            loss_contacts_dataset_orig.append(current_loss_contact_orig)
            loss_collision_n_points_dataset_orig.append(current_contact_n_points_orig)
            loss_collision_sum_depths_dataset_orig.append(current_contact_sum_depths_orig)

            current_loss_non_coll_optim = conglo_data.loc[idx, [follow_up_column + "_non_collision_optim"]].values[0]
            current_loss_contact_optim = conglo_data.loc[idx, [follow_up_column + "_contact_sample_optim"]].values[0]
            current_contact_n_points_optim = conglo_data.loc[idx, [follow_up_column + "_collision_points_optim"]].values[0]
            current_contact_sum_depths_optim = conglo_data.loc[idx, [follow_up_column + "_collision_sum_depths_optim"]].values[0]

            loss_non_collision_env_optim.append(current_loss_non_coll_optim)
            loss_contact_env_optim.append(current_loss_contact_optim)
            loss_collision_n_points_optim.append(current_contact_n_points_optim)
            loss_collision_sum_depths_optim.append(current_contact_sum_depths_optim)

            loss_non_collisions_dataset_optim.append(current_loss_non_coll_optim)
            loss_contacts_dataset_optim.append(current_loss_contact_optim)
            loss_collision_n_points_dataset_optim.append(current_contact_n_points_optim)
            loss_collision_sum_depths_dataset_optim.append(current_contact_sum_depths_optim)


        tb_data_orig.append([model, opt.dataset, current_env_name,
                        np.mean(loss_non_collision_env_orig),
                        np.std(loss_non_collision_env_orig),
                        np.mean(loss_contact_env_orig),
                        np.std(loss_contact_env_orig),
                        np.mean(loss_collision_n_points_orig),
                        np.std(loss_collision_n_points_orig),
                        np.mean(loss_collision_sum_depths_orig),
                        np.std(loss_collision_sum_depths_orig)
                        ])
        tb_data_optim.append([model, opt.dataset, current_env_name,
                              np.mean(loss_non_collision_env_optim),
                              np.std(loss_non_collision_env_optim),
                              np.mean(loss_contact_env_optim),
                              np.std(loss_contact_env_optim),
                              np.mean(loss_collision_n_points_optim),
                              np.std(loss_collision_n_points_optim),
                              np.mean(loss_collision_sum_depths_optim),
                              np.std(loss_collision_sum_depths_optim)
                              ])

    tb_data_orig.append([model, opt.dataset, "Overall",
                    np.mean(loss_non_collisions_dataset_orig),
                    np.std(loss_non_collisions_dataset_orig),
                    np.mean(loss_contacts_dataset_orig),
                    np.std(loss_contacts_dataset_orig),
                    np.mean(loss_collision_n_points_dataset_orig),
                    np.std(loss_collision_n_points_dataset_orig),
                    np.mean(loss_collision_sum_depths_dataset_orig),
                    np.std(loss_collision_sum_depths_dataset_orig)])

    tb_data_optim.append([model, opt.dataset, "Overall",
                          np.mean(loss_non_collisions_dataset_optim),
                          np.std(loss_non_collisions_dataset_optim),
                          np.mean(loss_contacts_dataset_optim),
                          np.std(loss_contacts_dataset_optim),
                          np.mean(loss_collision_n_points_dataset_optim),
                          np.std(loss_collision_n_points_dataset_optim),
                          np.mean(loss_collision_sum_depths_dataset_optim),
                          np.std(loss_collision_sum_depths_dataset_optim)])

    import logging

    logging.basicConfig(filename=f"output_{follow_up_column}_{opt.dataset}.txt", level=logging.INFO, format='')

    logging.info('\n\n\n')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n IT CLEARANCE ++ OUTPUT')
    logging.info('\n'+tabulate(tb_data_orig,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
    logging.info('\n'+tabulate(tb_data_orig,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))

    logging.info('\n\n\n')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n ---------------------------------------------------------------------------------------------')
    logging.info('\n OPTIMIZED IT CLEARANCE ++ OUTPUT')
    logging.info('\n'+tabulate(tb_data_optim,headers=tb_headers_optim, floatfmt=".4f",  tablefmt="latex_booktabs"))
    logging.info('\n'+tabulate(tb_data_optim,headers=tb_headers_optim, floatfmt=".4f",  tablefmt="simple"))

