import gc
import json
import os
import random
from os.path import join as opj
from shutil import copyfile

import pandas as pd
import numpy as np
import trimesh
import vedo
import warnings
import it
from it import util
from util.util_mesh import read_sdf
from util.util_proxd import load_smplx_model, load_vposer_model, optimize_body_on_environment
from utils import get_contact_id

if __name__ == '__main__':

    warnings.simplefilter("ignore", UserWarning)

    visualize= True
    save_results = False

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson"


    directory_datasets = opj(base_dir, "datasets")

    descriptors_dir = opj(base_dir, "config", "descriptors_repository")

    smplx_model_dir = opj(base_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_dir = opj(base_dir, "pretrained_place", "body_models", "vposer_v1_0")

    it_results_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'env_test')
    samples_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'sampled_it_clearance_opti_smplx')


    place_samples_dir = opj(base_dir,'test_place_picker[demo_conf]', 'sampled_place_exec')
    it_samples_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'sampled_it_clearance')

    vposer_model = load_vposer_model(vposer_model_dir)

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



    follow_up_file = opj(base_dir,'test_place_picker[demo_conf]', 'follow_up_process.csv')
    previous_follow_up_column = "num_it_picked_sampled"
    current_follow_up_column = "num_it_picked_sampled_opti_smplx"


    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = 0

    comb_dataset_escene = list(follow_up_data[ (follow_up_data[current_follow_up_column] < follow_up_data[previous_follow_up_column] )].index)
    pending_tasks = []
    for dataset_name, scene_name in comb_dataset_escene:
        final = follow_up_data.at[(dataset_name, scene_name), previous_follow_up_column]
        initial = follow_up_data.at[(dataset_name, scene_name), current_follow_up_column]
        for num_point in range(initial, final):
            pending_tasks.append((dataset_name, scene_name, num_point))

    num_pending_tasks = len(pending_tasks)
    num_total_task = follow_up_data['goal_place_picked_sampled'].sum()
    num_completed_task = num_total_task - num_pending_tasks
    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))


    last_env_used=None

    for current_dataset_name, current_scene_name, current_num_point in pending_tasks:
        print(current_dataset_name, current_scene_name, current_num_point)

        # extracting information about type of interaction on PLACE execution
        interaction_type_df = pd.read_csv(opj(place_samples_dir, current_scene_name, "interactions.txt"), index_col=0, header=None)
        interaction_type = interaction_type_df.at[current_num_point, 1]
        for current_interaction in interactions_by_type[interaction_type]:
            print(current_dataset_name, current_scene_name, current_num_point, current_interaction)

            directory_bodies = opj(it_samples_dir, current_scene_name)
            it_body_file = opj(directory_bodies, f"body_{current_num_point}_{current_interaction}.ply")
            if(  not os.path.exists(it_body_file) ):
                print(f"WARNING: no {current_interaction} found in point {current_num_point}")
                continue

            it_body = trimesh.load(it_body_file)

            file_mesh_env = opj(directory_datasets, current_dataset_name, "scenes", current_scene_name + ".ply")

            if last_env_used != current_scene_name:
                # get enviroment
                env_trimesh = trimesh.load(opj(directory_datasets, current_dataset_name, "scenes", f"{current_scene_name}.ply"))
                # get sdf environment information
                s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_sdf(opj(directory_datasets, current_dataset_name), current_scene_name)
                last_env_used = current_scene_name

            # get body "gender" and "contact regions"
            json_descriptor_file = [f for f in os.listdir(opj(descriptors_dir, current_interaction)) if f.endswith(".json")][0]
            with open(opj(descriptors_dir, current_interaction, json_descriptor_file)) as jsonfile:
                descriptor_data = json.load(jsonfile)
            contact_regions = descriptor_data["extra"]["contact_regions"]
            body_gender = descriptor_data["extra"]["body_gender"]

            # get body params
            body_params_file = [f for f in os.listdir(opj(descriptors_dir, current_interaction)) if f.endswith("_smplx_body_params.npy")][0]
            np_body_params = np.load(opj(descriptors_dir, current_interaction, body_params_file))

            smplx_model = load_smplx_model(smplx_model_dir, body_gender)
            last_gender = body_gender

            np_point = np.load(opj(samples_dir, current_scene_name,  f"point_{current_num_point}_{current_interaction}.npy"))
            np_best_angle = np.load(opj(samples_dir, current_scene_name, f"angle_{current_num_point}_{current_interaction}.npy"))

            contact_ids, _ = get_contact_id(body_segments_folder=opj(directory_datasets, "prox", 'body_segments'),
                                            contact_body_parts=contact_regions)

            body_trimesh_optim, np_body_params_optim = optimize_body_on_environment(
                env_trimesh, s_grid_min_batch, s_grid_max_batch, s_sdf_batch,
                smplx_model, vposer_model,
                np_body_params, np_point, np_best_angle, contact_ids,
                weight_loss_rec_verts=1.0,
                weight_loss_rec_bps=1.0,
                weight_loss_vposer=0.02,
                weight_loss_shape=0.01,
                weight_loss_hand=0.01,
                weight_collision=8.0,
                weight_loss_contact=0.5,
                itr_s2=150,
                view_evolution_screens=visualize)

            if visualize:
                s = trimesh.Scene()
                s.add_geometry(env_trimesh)
                s.add_geometry(body_trimesh_optim)
                s.show()

            if save_results:
                output_subdir = opj(output_dir, current_scene_name)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                it_body.export(opj(output_subdir, f"body_{current_num_point}_{current_interaction}.ply"))

        if save_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(current_dataset_name, current_scene_name), current_follow_up_column] = current_num_point + 1
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
        gc.collect()