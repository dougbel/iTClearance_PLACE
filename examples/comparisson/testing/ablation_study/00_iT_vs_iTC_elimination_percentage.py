"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import json
import os
from os.path import  join as opj

from trimesh import load_mesh

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores
from si.fulldatascores import FullDataScores
from util.util_mesh import find_files_mesh_env
import pandas as pd

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")

    with_fillers = False  # True   False                  This indicates if use data generated iTClearance in environments with fillers

    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")


    if with_fillers:
        env_test_dir = opj(base_dir, "test", "env_test")
        output_dir = opj(output_base, "bubble_fillers")
    else:
        env_test_dir = opj(base_dir, "test", "no_filled_env_test")
        output_dir = opj(output_base, "no_bubble_fillers")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    datasets_dir = opj(base_dir, "datasets")

    results= pd.DataFrame(columns=["dataset", "scene", "interaction", "detected_iT", "detected_iTC"])

    for scene  in os.listdir(env_test_dir):
        if not os.path.isdir(opj(env_test_dir, scene)):
            continue
        file_mesh_env, dataset_name = find_files_mesh_env(datasets_dir, scene)
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
            clearance_np_points, clearance_np_scores, clearance_np_missings, clearance_np_cv_collided = scores_data.filter_data_scores(
                max_limit_score,
                max_limit_missing,
                max_limit_cv_collided)


            naive_score_data = FullDataScores(df_scores_data, interaction)
            naive_np_points, naive_np_scores, naive_np_missings = naive_score_data.filter_data_scores(max_limit_score, max_limit_missing)



            results.loc[len(results.index)] = [dataset_name, scene, interaction, len(naive_np_points),len(clearance_np_points)]

    results.to_csv(opj(output_dir,"00_iT_vs_iTC_elimination_percentage.csv"))