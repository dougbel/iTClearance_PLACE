"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import warnings

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import pandas as pd
import math




if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")

    id = "itClearance"  # None


    n_sample_per_scene=97 # confidence level = 95%, margin error = 10%  for infinite samples

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")

    conglo_path= opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv")


    column_prefix = f"piloting_"
    conglo_data = pd.read_csv(conglo_path)

    # just for keeping same samples
    conglo_data = conglo_data[conglo_data["comparison_itClearance"]==True]


    follow_up_column = f"{column_prefix}{id}"
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

    filtered_conglo_data = conglo_data[conglo_data[follow_up_column]==True]

    sampling_file_path= opj(env_filled_data_test_dir, "piloting", follow_up_column+"_template.csv")
    filtered_conglo_data.to_csv(sampling_file_path,index=False)