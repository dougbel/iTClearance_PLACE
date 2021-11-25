"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import json
import os
from os.path import join as opj

import pandas as pd

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores
from si.fulldatascores import FullDataScores
from util.util_mesh import find_files_mesh_env

interactions_by_type = {
        "laying": ["laying_bed", "laying_hands_up", "laying_on_sofa", "laying_sofa_foot_on_floor"],
        "reaching_out": ["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table",
                         "reaching_out_ontable_one_hand"],
        "sitting": ["sitting", "sitting_bit_open_arms", "sitting_chair", "sitting_comfortable", "sitting_compact",
                    "sitting_hands_on_device", "sitting_looking_to_right", "sitting_small_table", "sitting_stool",
                    "sitting_stool_one_foot_floor"],
        "standing_up": ["standing_up", "standup_hand_on_furniture"],
        "walking": ["walking_left_foot", "walking_right_foot"]
    }


def get_interaction_type(interaction_name):
    for current_type in interactions_by_type:
        if interaction_name in interactions_by_type[current_type]:
            return current_type

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")

    with_fillers = True # True   False                  This indicates if use data generated iTClearance in environments with fillers
    filter_dataset = None    # None   prox   mp3d  replica_v1

    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    if with_fillers:
        env_test_dir = opj(base_dir, "test", "env_test")
        output_dir = opj(output_base, "bubble_fillers")
    else:
        env_test_dir = opj(base_dir, "test", "no_filled_env_test")
        output_dir = opj(output_base, "no_bubble_fillers")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    conglomerator_naive= None
    conglomerator_clearance= None

    for scene  in os.listdir(env_test_dir):
        if not os.path.isdir(opj(env_test_dir, scene)):
            continue
        file_mesh_env, dataset_name = find_files_mesh_env(datasets_dir, scene)

        if filter_dataset is not None  and dataset_name !=filter_dataset:
            continue

        # print(file_mesh_env)
        for interaction in os.listdir(opj(env_test_dir, scene)):
            env_test_results_dir = opj(env_test_dir, scene, interaction)
            json_conf_execution_file = opj(json_conf_execution_dir, f"single_testing_{interaction}.json")

            # print(env_test_results_dir)
            # print(json_conf_execution_file)

            tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
            subdir_name = "_".join(tester.affordances[0])
            propagation_settings_file = os.path.join(directory_of_prop_configs, subdir_name, 'propagation_data.json')
            with open(propagation_settings_file) as json_file:
                propagation_settings = json.load(json_file)
            max_limit_score = propagation_settings['max_limit_score']
            max_limit_missing = propagation_settings['max_limit_missing']
            max_limit_cv_collided = propagation_settings['max_limit_cv_collided']

            df_scores_data = pd.read_csv(os.path.join(env_test_results_dir, "test_scores.csv"))
            scores_data = FullDataClearanceScores(df_scores_data, interaction)
            sub_df = scores_data.filter_dataframe_best_score_per_point( max_limit_score, max_limit_missing, max_limit_cv_collided)
            sub_df.insert(loc=0, column='interaction_type', value=get_interaction_type(interaction))
            sub_df.insert(loc=0, column='interaction', value=interaction)
            sub_df.insert(loc=0, column='scene', value=scene)
            sub_df.insert(loc=0, column='dataset', value=dataset_name)


            naive_score_data = FullDataScores(df_scores_data, interaction)
            naive_sub_df = naive_score_data.filter_dataframe_best_score_per_point(max_limit_score, max_limit_missing)
            naive_sub_df.insert(loc=0, column='interaction_type', value=get_interaction_type(interaction))
            naive_sub_df.insert(loc=0, column='interaction', value=interaction)
            naive_sub_df.insert(loc=0, column='scene', value=scene)
            naive_sub_df.insert(loc=0, column='dataset', value=dataset_name)

            if conglomerator_naive is None:
                conglomerator_clearance = pd.DataFrame(columns=sub_df.columns.values.tolist())
                conglomerator_naive = pd.DataFrame(columns=sub_df.columns.values.tolist())

            conglomerator_clearance = conglomerator_clearance.append(sub_df)
            conglomerator_naive = conglomerator_naive.append(naive_sub_df)

    conglomerator_clearance.sort_values(["dataset", "scene", "interaction"], inplace=True)
    conglomerator_naive.sort_values(["dataset", "scene", "interaction"], inplace=True)
    if filter_dataset is None:
        conglomerator_clearance.to_csv(opj(output_dir,"02_conglomerate_capable_positions_clearance.csv"), index=False)
        conglomerator_naive.to_csv(opj(output_dir, "02_conglomerate_capable_positions_naive.csv"), index=False)
    else:
        conglomerator_clearance.to_csv(opj(output_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"), index=False)
        conglomerator_naive.to_csv(opj(output_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"), index=False)