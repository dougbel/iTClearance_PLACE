"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import statistics
import warnings

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch

import pandas as pd
import numpy as np

from tabulate import tabulate


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")

    stratified_sampling = True

    id = "itClearance"  # None

    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filles_to_test={
        # "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive.csv"),
        # "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv"),
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"full_02_conglomerate_capable_positions_clearance.csv")
    }


    column_prefix = f"comparison_"

    for model in filles_to_test:

        tb_headers = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact", "contact_std",
                      "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
        tb_headers_optim = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact", "contact_std",
                      "collision_points", "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
        tb_data = []
        tb_data_optim = []

        conglo_path =filles_to_test[model]
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
                                np.mean(loss_non_collision_env),
                                np.std(loss_non_collision_env),
                                np.mean(loss_contact_env),
                                np.std(loss_contact_env),
                                np.mean(loss_collision_n_points),
                                np.std(loss_collision_n_points),
                                np.mean(loss_collision_sum_depths),
                                np.std(loss_collision_sum_depths)
                                ])
                tb_data_optim.append([model, dataset_name, current_env_name,
                                      np.mean(loss_non_collision_env_optim),
                                      np.std(loss_non_collision_env_optim),
                                      np.mean(loss_contact_env_optim),
                                      np.std(loss_contact_env_optim),
                                      np.mean(loss_collision_n_points_optim),
                                      np.std(loss_collision_n_points_optim),
                                      np.mean(loss_collision_sum_depths_optim),
                                      np.std(loss_collision_sum_depths_optim)
                                      ])

            tb_data.append([model, dataset_name, "Overall",
                            np.mean(loss_non_collisions_dataset),
                            np.std(loss_non_collisions_dataset),
                            np.mean(loss_contacts_dataset),
                            np.std(loss_contacts_dataset),
                            np.mean(loss_collision_n_points_dataset),
                            np.std(loss_collision_n_points_dataset),
                            np.mean(loss_collision_sum_depths_dataset),
                            np.std(loss_collision_sum_depths_dataset)])

            tb_data_optim.append([model, dataset_name, "Overall",
                                  np.mean(loss_non_collisions_dataset_optim),
                                  np.std(loss_non_collisions_dataset_optim),
                                  np.mean(loss_contacts_dataset_optim),
                                  np.std(loss_contacts_dataset_optim),
                                  np.mean(loss_collision_n_points_dataset_optim),
                                  np.std(loss_collision_n_points_dataset_optim),
                                  np.mean(loss_collision_sum_depths_dataset_optim),
                                  np.std(loss_collision_sum_depths_dataset_optim)])

        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.DEBUG, format='')

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
