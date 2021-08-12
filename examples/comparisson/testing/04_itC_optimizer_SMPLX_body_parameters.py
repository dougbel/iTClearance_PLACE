import argparse
import gc
import json
import os
import random
import warnings
from os.path import join as opj
from shutil import copyfile

import pandas as pd
import numpy as np
import trimesh

from util.util_mesh import read_sdf
from util.util_proxd import optimize_body_on_environment, load_smplx_model, load_vposer_model
from utils import get_contact_id

warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, help='Information directory (dataset, pretrained models, etc)')
opt = parser.parse_args()
print(opt)


if __name__ == '__main__':

    visualize = True
    shuffle_order = False  # if shuffle is True then execution would be SLOWER
    save_results = False

    base_dir = opt.base_dir

    datasets_dir = opj(base_dir, "datasets")

    descriptors_dir = opj(base_dir, "config", "descriptors_repository")

    smplx_model_dir = opj(base_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_dir = opj(base_dir, "pretrained_place", "body_models", "vposer_v1_0")

    it_results_dir = opj(base_dir, 'test', 'env_test')
    samples_dir = opj(base_dir, 'test', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test', 'sampled_it_clearance_opti_smplx')

    follow_up_file = opj(base_dir, 'test', 'follow_up_process.csv')
    previus_follow_up_column = "it_auto_samples"
    current_follow_up_column = "it_auto_samples_opti_smplx"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if current_follow_up_column not in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[(follow_up_data[current_follow_up_column] == False)
                                        & (follow_up_data[previus_follow_up_column] == True)].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[follow_up_data[current_follow_up_column] == True].index.size

    print('STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    if shuffle_order:
        random.shuffle(pending_tasks)

    vposer_model = load_vposer_model(vposer_model_dir)

    last_env = None
    last_gender = None
    last_interaction = None

    for dataset, env_name, interaction in pending_tasks:
        print(dataset, env_name, interaction)

        if last_env != env_name:
            # get enviroment
            env_trimesh = trimesh.load(opj(datasets_dir, dataset, "scenes", f"{env_name}.ply"))
            # get sdf environment information
            s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_sdf(opj(datasets_dir, dataset), env_name)
            last_env = env_name


        if last_interaction != interaction:
            #get body "gender" and "contact regions"
            json_descriptor_file = [f for f in os.listdir(opj(descriptors_dir, interaction)) if f.endswith(".json")][0]
            with open( opj(descriptors_dir, interaction, json_descriptor_file) ) as jsonfile:
                descriptor_data = json.load(jsonfile)
            contact_regions = descriptor_data["extra"]["contact_regions"]
            body_gender = descriptor_data["extra"]["body_gender"]

            #get body params
            body_params_file = [f for f in os.listdir(opj(descriptors_dir, interaction)) if f.endswith("_smplx_body_params.npy")][0]
            np_body_params = np.load(opj(descriptors_dir, interaction,body_params_file))

            last_interaction = interaction


        if last_gender != body_gender:
            smplx_model = load_smplx_model(smplx_model_dir, body_gender)
            last_gender = body_gender

        for i in range(follow_up_data.loc[dataset, env_name, interaction]['num_it_auto_samples']):

            np_point = np.load(opj(samples_dir,env_name, interaction, f"point_{i}.npy"))
            np_best_angle = np.load(opj(samples_dir,env_name, interaction, f"angle_{i}.npy"))

            contact_ids, _ = get_contact_id(body_segments_folder=opj(datasets_dir, "prox", 'body_segments'),
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
                output_subdir = opj(output_dir, env_name, interaction)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                body_trimesh_optim.export(opj(output_subdir, f"body_{i}.ply"))
                np.save(opj(output_subdir, f"body_{i}_smplx_body_params.npy"), np_body_params_optim)

        if save_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset, env_name, interaction), current_follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")

        gc.collect()
