"""
Same sampling positions would be use for the PLACE execution to have a fair comparison over the same elements in
the environment
"""
import warnings

from numpy.distutils.command.config import config

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

    output_file_name_pattern = "02_conglomerate_capable_positions_clearance_PLACE_tests"

    idiT = "itClearance"
    id = "PLACE"  # None

    visualize = True


    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    filles_to_test={
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv")
    }


    column_prefix = f"comparison_"

    for model in filles_to_test:

        conglo_path =filles_to_test[model]
        print(conglo_path)

        conglo_data = pd.read_csv(conglo_path)

        conglo_data = conglo_data[conglo_data[f"{column_prefix}{idiT}"] ==True]
        for column in conglo_data.columns:
            if column.startswith(f"{column_prefix}{idiT}_"):
                del conglo_data[column]

        if id is None:
            n_sampling = get_next_sampling_id(conglo_data.columns.to_list())
        else:
            n_sampling = id

        follow_up_column = f"{column_prefix}{n_sampling}"
        conglo_data[follow_up_column] = True
        conglo_data[follow_up_column + "_non_collision"] = ""
        conglo_data[follow_up_column + "_contact_sample"] = ""
        conglo_data[follow_up_column + "_collision_points"] = ""
        conglo_data[follow_up_column + "_collision_sum_depths"] = ""
        conglo_data[follow_up_column + "_non_collision_optim"] = ""
        conglo_data[follow_up_column + "_contact_sample_optim"] = ""
        conglo_data[follow_up_column + "_collision_points_optim"] = ""
        conglo_data[follow_up_column + "_collision_sum_depths_optim"] = ""
        conglo_data[follow_up_column + "_non_collision_adv_optim"] = ""
        conglo_data[follow_up_column + "_contact_sample_adv_optim"] = ""
        conglo_data[follow_up_column + "_collision_points_adv_optim"] = ""
        conglo_data[follow_up_column + "_collision_sum_depths_adv_optim"] = ""

        n_sample_per_scene = conglo_data['scene'].value_counts().values[0]

        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.INFO, format='')
        logging.info(f"Sampling: per dataset same number interaction on positions selected by iTClearance to a fair comparisson")
        logging.info(f"n_sample_per_scene:  {n_sample_per_scene}")

        grouped = conglo_data.groupby('scene')
        for current_env_name in conglo_data['scene'].unique():
            per_scene = grouped.get_group(current_env_name)
            per_scene.to_csv(opj(env_filled_data_test_dir, f"{output_file_name_pattern}_{current_env_name}.csv"),index=False)