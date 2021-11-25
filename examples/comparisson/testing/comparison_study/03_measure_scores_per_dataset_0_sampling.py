"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import warnings

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import pandas as pd
import math


def get_next_sampling_id(l_column_names):
    return len([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1



if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")


    id = "itClearance"  # None

    visualize = True

    n_sample_per_scene=1297 # confidence level = 97%, margin error = 3%  for infinite samples
    # n_sample_per_scene=10 #


    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")


    filles_to_test={
        # "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive.csv"),
        # "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv"),
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv")
    }


    column_prefix = f"comparison_"

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
        conglo_data[follow_up_column + "_non_collision"] = ""
        conglo_data[follow_up_column + "_contact_sample"] = ""
        conglo_data[follow_up_column + "_collision_points"] = ""
        conglo_data[follow_up_column + "_collision_sum_depths"] = ""
        conglo_data[follow_up_column + "_non_collision_optim"] = ""
        conglo_data[follow_up_column + "_contact_sample_optim"] = ""
        conglo_data[follow_up_column + "_collision_points_optim"] = ""
        conglo_data[follow_up_column + "_collision_sum_depths_optim"] = ""

        grouped = conglo_data.groupby('scene')

        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.INFO, format='')
        logging.info(f"Sampling: per dataset same number interaction type, simple random sampling in each interaction type")
        logging.info(f"n_sample_per_scene:  {n_sample_per_scene}")

        for current_env_name in conglo_data['scene'].unique():

            dataset_results = grouped.get_group(current_env_name)

            per_inter_type = math.ceil(n_sample_per_scene / len(dataset_results['interaction_type'].unique()))

            sample = dataset_results.groupby('interaction_type', group_keys=False).apply(lambda x: x.sample(per_inter_type)).sample(frac=1)

            conglo_data.loc[sample.index.to_list(),[follow_up_column]] =True

            logging.info(f"current_env_name:  {current_env_name},\t samples: {len(sample)}")


        conglo_data.to_csv(conglo_path,index=False)