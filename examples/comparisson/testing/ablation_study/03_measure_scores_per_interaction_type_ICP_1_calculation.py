"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import os
import statistics
import warnings

import trimesh
import vedo

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from vedo import load
from tqdm import tqdm

from it_clearance.testing.tester import TesterClearance
from util.util_mesh import read_sdf, find_files_mesh_env
import pandas as pd
import numpy as np
import torch.nn.functional as F
from it import util

import gc

from tabulate import tabulate

def get_next_sampling_id(l_column_names):
    return len([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1


def measure_trimesh_collision(trimesh_decimated_env, it_body):
    influence_radio_bb = 1.5
    extension, middle_point = util.influence_sphere(it_body, influence_radio_bb)
    tri_mesh_env_cropped = util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point, extension)

    collision_tester = trimesh.collision.CollisionManager()
    collision_tester.add_object('env', tri_mesh_env_cropped)
    in_collision, contact_data = collision_tester.in_collision_single(it_body, return_data=True)

    return contact_data

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")

    # stratified_sampling = True

    # n_sample_per_interaction_type=382 # confidence level = 95%, margin error = 5%  for infinite samples
    n_sample_per_interaction_type=1297 # confidence level = 97%, margin error = 3%  for infinite samples
    # n_sample_per_interaction_type=2 #

    filter_dataset = "prox"    # None   prox   mp3d  replica_v1

    id = "icpst"  # None   # if None it choose the next numerical id

    visualize = False

    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filles_to_test={
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv")
    }


    column_prefix = f"ablation_{filter_dataset}_"
    for model in filles_to_test:
        gc.collect()
        torch.cuda.empty_cache()
        conglo_path =filles_to_test[model]
        print(conglo_path)

        conglo_data = pd.read_csv(conglo_path)

        if id is None:
            n_sampling = get_next_sampling_id(conglo_data.columns.to_list())
        else:
            n_sampling = id

        follow_up_column = f"{column_prefix}{n_sampling}"

        grouped = conglo_data.groupby(conglo_data['interaction_type'])

        ##########################################################################
        ######  Shortcut for collision manager calculation
        decimated_envs = {}
        for scene in conglo_data['scene'].unique():
            file_mesh_env, __ = find_files_mesh_env(datasets_dir, scene)
            decimated_envs[scene] = vedo.vtk2trimesh(vedo.load(file_mesh_env).decimate(fraction=.3))
            # decimated_envs[scene] = trimesh.load_mesh(file_mesh_env)
        ##########################################################################

        print(conglo_path)

        for current_interaction_type in conglo_data['interaction_type'].unique():
            gc.collect()
            torch.cuda.empty_cache()

            interaction_type_results = grouped.get_group(current_interaction_type)

            sample = interaction_type_results[ interaction_type_results[follow_up_column] == True ]
            # sample[follow_up_column + "non_collision"] = 0.0
            # sample[follow_up_column + "contact_sample"] = 0.0
            # sample[follow_up_column + "collision_points"] = 0.0
            # sample[follow_up_column + "collision_sum_depths"] = 0.0

            for idx, row in tqdm(sample.iterrows(), total=sample.shape[0]):

                if sample.loc[idx, [follow_up_column + "non_collision"]].isnull().values[0] == False:
                    continue

                gc.collect()
                torch.cuda.empty_cache()

                dataset = row["dataset"]
                env_name = row["scene"]
                interaction = row["interaction"]
                angle = row["angle"]
                json_conf_execution_file = opj(json_conf_execution_dir, f"single_testing_{interaction}.json")
                tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
                subdir_name = "_".join(tester.affordances[0])
                ply_obj_file = opj(directory_of_trainings, interaction, subdir_name + "_object.ply")
                vtk_object = load(ply_obj_file)
                vtk_object.rotate(angle, axis=(0, 0, 1), rad=True)
                vtk_object.pos(x=row["point_x"], y=row["point_y"], z=row["point_z"])
                trimesh_obj = vedo.vtk2trimesh(vtk_object)


                trimesh_decimated_env =  decimated_envs[env_name]

                influence_radio_bb = 1.25
                extension, middle_point = util.influence_sphere(trimesh_obj, influence_radio_bb)
                tri_mesh_env_cropped = util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point,
                                                                          extension)

                matrix, transformation, cost = trimesh.registration.icp(trimesh_obj.vertices, tri_mesh_env_cropped,
                                                                        max_iterations=3,
                                                                        reflection=False, scale=False)
                trimesh_translated_obj = trimesh.Trimesh(vertices=transformation, faces=trimesh_obj.faces)

                contact_data = measure_trimesh_collision(trimesh_decimated_env, trimesh_translated_obj)

                if visualize:
                    s = trimesh.Scene()
                    it_body_orig = vedo.vtk2trimesh(vtk_object)
                    it_body_orig.visual.face_colors = [200, 200, 200, 150]
                    trimesh_translated_obj.visual.face_colors = [200, 200, 200, 255]
                    s.add_geometry(it_body_orig)
                    s.add_geometry(trimesh_translated_obj)
                    s.add_geometry(trimesh_decimated_env)
                    s.show()

                vtk_object = vedo.trimesh2vtk(trimesh_translated_obj)


                s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_sdf(opj(datasets_dir, dataset), env_name)

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
                    current_loss_non_coll= 1.0
                    current_loss_contact = 0.0
                else:
                    current_loss_non_coll = (body_sdf_batch > 0).sum().float().item() / 10475.0
                    current_loss_contact = 1.0

                current_contact_n_points = len(contact_data)
                current_contact_sum_depths = sum([data.depth for data in contact_data])

                conglo_data.loc[idx,[follow_up_column + "non_collision"]] = current_loss_non_coll
                conglo_data.loc[idx,[follow_up_column + "contact_sample"]] = current_loss_contact
                conglo_data.loc[idx,[follow_up_column + "collision_points"]] = current_contact_n_points
                conglo_data.loc[idx,[follow_up_column + "collision_sum_depths"]] = current_contact_sum_depths


                conglo_data.to_csv(conglo_path,index=False)