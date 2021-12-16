"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import argparse
import gc
import json
import os
import statistics
import warnings

import trimesh

from it import util
from util.util_proxd import load_smplx_model, optimize_body_on_environment, load_vposer_model
from utils import get_contact_id

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from vedo import load, trimesh2vtk, vtk2trimesh

from it_clearance.testing.tester import TesterClearance
from util.util_mesh import read_sdf, find_files_mesh_env
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from tabulate import tabulate



def get_next_sampling_id(l_column_names):
    return len([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1


def measure_scores(vtk_object, s_sdf_batch, s_grid_min_batch, s_grid_max_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################  compute non-collision/contact score ##############
    # body verts before optimization
    # [1, 10475, 3]
    body_verts_sample = np.asarray(vtk_object.points())
    body_verts_sample_prox_tensor = torch.from_numpy(body_verts_sample).float().unsqueeze(0).to(device)
    norm_verts_batch = (body_verts_sample_prox_tensor - s_grid_min_batch) / (
            s_grid_max_batch - s_grid_min_batch) * 2 - 1
    body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                   norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                   padding_mode='border')

    current_loss_non_coll = 0.0
    current_loss_contact = 0.0
    if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
        current_loss_non_coll = 1.0
        current_loss_contact = 0.0
    else:
        current_loss_non_coll = (body_sdf_batch > 0).sum().float().item() / 10475.0
        current_loss_contact = 1.0

    return current_loss_non_coll, current_loss_contact

def measure_trimesh_collision(trimesh_decimated_env, it_body):
    influence_radio_bb = 1.5
    extension, middle_point = util.influence_sphere(it_body, influence_radio_bb)
    tri_mesh_env_cropped = util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point, extension)

    collision_tester = trimesh.collision.CollisionManager()
    collision_tester.add_object('env', tri_mesh_env_cropped)
    in_collision, contact_data = collision_tester.in_collision_single(it_body, return_data=True)

    return contact_data



parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scene', required=True, help='scene')
opt = parser.parse_args()
print(opt)


if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "comparison_study_test")

    stratified_sampling = True

    id = "itClearance"  # None

    visualize = False

    # n_sample_per_scene=1297 # confidence level = 97%, margin error = 3%  for infinite samples


    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    smplx_model_dir = opj(base_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_dir = opj(base_dir, "pretrained_place", "body_models", "vposer_v1_0")
    vposer_model = load_vposer_model(vposer_model_dir)

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Executing on device " + str(device))

    column_prefix = f"comparison_"

    current_env_name = opt.scene

    output_path = opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{current_env_name}.csv")
    if os.path.exists(output_path):
        conglo_path = output_path
    else:
        conglo_path = opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance.csv")
    print(conglo_path)
    print("output: ", output_path)


    conglo_data = pd.read_csv(conglo_path)
    old_follow_up_column = f"{column_prefix}{id}"
    follow_up_column = f"{column_prefix}{id}_improved"
    grouped = conglo_data.groupby('scene')
    sample = grouped.get_group(current_env_name)
    sample = sample[ sample[old_follow_up_column] == True ]


    dataset_path, dataset_name = find_files_mesh_env(datasets_dir, current_env_name)
    env_trimesh = trimesh.load(dataset_path)
    s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_sdf(opj(datasets_dir, dataset_name), current_env_name)

    print(f"{current_env_name}  -     {dataset_name}")


    if follow_up_column + "_non_collision" not in conglo_data.columns:
        sample[follow_up_column] = False
        sample[follow_up_column + "_non_collision"] = ""
        sample[follow_up_column + "_contact_sample"] = ""
        sample[follow_up_column + "_collision_points"] = ""
        sample[follow_up_column + "_collision_sum_depths"] = ""
        sample[follow_up_column + "_non_collision_optim"] = ""
        sample[follow_up_column + "_contact_sample_optim"] = ""
        sample[follow_up_column + "_collision_points_optim"] = ""
        sample[follow_up_column + "_collision_sum_depths_optim"] = ""


    counter=0
    for idx, row in tqdm(sample.iterrows(), total=sample.shape[0] ):
        if sample.loc[idx, [follow_up_column]].values[0] == True:
            continue

        dataset = row["dataset"]
        env_name = row["scene"]
        interaction = row["interaction"]
        angle = row["angle"]
        point = np.array([row["point_x"], row["point_y"], row["point_z"]])

        json_conf_execution_file = opj(json_conf_execution_dir, f"single_testing_{interaction}.json")
        tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
        subdir_name = "_".join(tester.affordances[0])
        ply_obj_file = opj(directory_of_trainings, interaction, subdir_name + "_object.ply")
        vtk_object = load(ply_obj_file)
        vtk_object.rotate(angle, axis=(0, 0, 1), rad=True)
        vtk_object.pos(x=row["point_x"], y=row["point_y"], z=row["point_z"])



        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # measuring for simple iTClearance++
        body_trimesh = vtk2trimesh(vtk_object)
        current_loss_non_coll, current_loss_contact = measure_scores(vtk_object, s_sdf_batch, s_grid_min_batch, s_grid_max_batch)
        contact_data = measure_trimesh_collision(env_trimesh, body_trimesh)
        current_contact_n_points = len(contact_data)
        current_contact_sum_depths = sum([data.depth for data in contact_data])

        sample.loc[idx, follow_up_column + "_non_collision"] = current_loss_non_coll
        sample.loc[idx, follow_up_column + "_contact_sample"] = current_loss_contact
        sample.loc[idx, follow_up_column + "_collision_points"] = current_contact_n_points
        sample.loc[idx, follow_up_column + "_collision_sum_depths"] = current_contact_sum_depths

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #measuring for optimized iTClearance++
        json_descriptor_file = [f for f in os.listdir(opj(directory_of_trainings, interaction)) if f.endswith(".json")][0]
        with open(opj(directory_of_trainings, interaction, json_descriptor_file)) as jsonfile:
            descriptor_data = json.load(jsonfile)
        contact_regions = descriptor_data["extra"]["contact_regions"]
        body_gender = descriptor_data["extra"]["body_gender"]
        body_params_file = [f for f in os.listdir(opj(directory_of_trainings, interaction)) if f.endswith("_smplx_body_params.npy")][0]
        np_body_params = np.load(opj(directory_of_trainings, interaction, body_params_file))
        contact_ids, _ = get_contact_id(body_segments_folder=opj(datasets_dir, "prox", 'body_segments'), contact_body_parts=contact_regions)
        smplx_model = load_smplx_model(smplx_model_dir, body_gender)

        if dataset_name == "replica_v1":
            weight_collision=0.15
            weight_loss_contact=0.50
        elif dataset_name == "mp3d":
            weight_collision = 6.0
            weight_loss_contact = 0.50
        else:
            weight_collision = 8.0
            weight_loss_contact = 0.50

        body_trimesh_optim, np_body_params_optim = optimize_body_on_environment(
            env_trimesh, s_grid_min_batch, s_grid_max_batch, s_sdf_batch,
            smplx_model, vposer_model,
            np_body_params, point, angle, contact_ids,
            weight_loss_rec_verts=1.0,
            weight_loss_rec_bps=1.0,
            weight_loss_vposer=0.10,
            weight_loss_shape=0.08,
            weight_loss_hand=0.01,
            weight_collision=weight_collision,
            weight_loss_contact=weight_loss_contact,
            itr_s2=150,
            view_evolution_screens=visualize)

        vtk_object_optim =  trimesh2vtk(body_trimesh_optim)
        current_loss_non_coll_optim, current_loss_contact_optim = measure_scores(vtk_object_optim, s_sdf_batch, s_grid_min_batch, s_grid_max_batch)
        contact_data_optim = measure_trimesh_collision(env_trimesh, body_trimesh_optim)
        current_contact_n_points_optim = len(contact_data_optim)
        current_contact_sum_depths_optim = sum([data.depth for data in contact_data_optim])

        sample.loc[idx, follow_up_column + "_non_collision_optim"] = current_loss_non_coll_optim
        sample.loc[idx, follow_up_column + "_contact_sample_optim"] = current_loss_contact_optim
        sample.loc[idx, follow_up_column + "_collision_points_optim"] = current_contact_n_points_optim
        sample.loc[idx, follow_up_column + "_collision_sum_depths_optim"] = current_contact_sum_depths_optim
        sample.loc[idx, follow_up_column] = True

        if visualize:
            json_conf_execution_file = opj(json_conf_execution_dir, f"single_testing_{interaction}.json")
            tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
            subdir_name = "_".join(tester.affordances[0])
            ply_obj_file = opj(directory_of_trainings, interaction, subdir_name + "_object.ply")

            body_trimesh.visual.face_colors = [200, 200, 200, 150]
            body_trimesh_optim.visual.face_colors = [200, 200, 200, 255]

            s = trimesh.Scene()
            s.add_geometry(env_trimesh)
            s.add_geometry(body_trimesh_optim)
            s.add_geometry(body_trimesh)
            s.show()


        #partial saving
        if counter % 20 == 0:
            print(f"env: {current_env_name}, counter: {counter}, saving data on csv {output_path} ")
            sample.to_csv(output_path,index=False)
            gc.collect()
            torch.cuda.empty_cache()
        counter += 1
    #final saving
    sample.to_csv(output_path, index=False)