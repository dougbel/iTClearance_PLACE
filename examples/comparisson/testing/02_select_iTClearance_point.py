import os
import json
from os.path import join as opj
from shutil import copyfile

import numpy as np
import pandas as pd
from vedo import vtk2trimesh
import trimesh

from ctrl.point_selection import ControlPointSelection
from util.util_mesh import find_files_mesh_env

if __name__ == '__main__':


    # [ 'reaching_out_mid_up', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_mid',
    # 'sitting_looking_to_right', 'sitting_compact', 'reachin_out_ontable_one_hand'
    # 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot']

    # interaction = 'reaching_out_mid_up'


    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    directory_descriptors = opj(base_dir, "config", "descriptors_repository")
    directory_json_conf_execution = opj(base_dir, "config", "json_execution")
    directory_of_prop_configs = opj(base_dir, "config", "propagators_configs")
    directory_datasets = opj(base_dir, "datasets")

    test_results_dir = opj(base_dir,'test', 'env_test')
    output_dir = opj(base_dir, 'test', 'picked_it_clearance')

    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    follow_up_column = "it_samples_extracted"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not follow_up_column in follow_up_data.columns:
        follow_up_data[follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[follow_up_data[follow_up_column] == False].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[follow_up_data[follow_up_column] == True].index.size

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    for dataset, env_name, interaction in pending_tasks:
    # for env_name in os.listdir(test_results_dir):
        directory_env_test = opj(test_results_dir, env_name)
        file_mesh_env, dataset_name = find_files_mesh_env(directory_datasets, env_name)

        # for interaction in os.listdir(directory_env_test):
        aff_res_test_dir = opj(directory_env_test, interaction)

        json_training_file = opj(aff_res_test_dir, "test_data.json")
        with open(json_training_file) as f:
            test_data = json.load(f)

        file_json_conf_execution = opj(directory_json_conf_execution, f"single_testing_{interaction}.json")
        scores_ctrl = ControlPointSelection(directory_descriptors, file_json_conf_execution,
                                    directory_env_test, directory_of_prop_configs, file_mesh_env)

        scores_ctrl.start()
        scores_ctrl.view.vp.close()

        for i in range(len( scores_ctrl.points_selected )):
            output_suddir = opj(output_dir, env_name, interaction)
            if not os.path.exists(output_suddir):
                os.makedirs(output_suddir)

            np.save(opj(output_suddir, f"point_{i}"), scores_ctrl.points_selected[i])
            vtk2trimesh(scores_ctrl.vedo_best_bodies[i]).export(opj(output_suddir,f"body_{i}.ply"))

        if len( scores_ctrl.points_selected ) >= 3:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset, env_name, interaction), follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
