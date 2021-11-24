"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import gc
import os
import statistics
import warnings

import trimesh
import vedo
from tqdm import tqdm

from it import util

warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from vedo import load

from it_clearance.testing.tester import TesterClearance
from util.util_mesh import read_sdf, find_files_mesh_env
import pandas as pd
import numpy as np
import torch.nn.functional as F

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

    stratified_sampling = False
    fixed_n_sampling = None # missing 3 and 5 you can choose the number of experiment sampling to perform AND/OR override in csv file

    # n_sample_per_scene=382 # confidence level = 95%, margin error = 5%  for infinite samples
    # n_sample_per_interaction_type=1297 # confidence level = 97%, margin error = 3%  for infinite samples
    n_sample_per_interaction_type=5 #

    visualize=False

    filter_dataset = "prox"    # None   prox   mp3d  replica_v1

    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filles_to_test={
        "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv")
    }


    column_prefix = f"ablation_{filter_dataset}_"
    for model in filles_to_test:
        tb_headers = ["model", "dataset", "interaction_type", "non_collision", "std_dev", "contact", "collision_points",
                      "collision_points_std_dev", "collision_depth", "collision_depth_std_dev"]
        tb_data = []
        conglo_path =filles_to_test[model]
        print(conglo_path)

        loss_non_collisions_model=[]
        loss_contacts_model=[]
        loss_collision_n_points_model = []
        loss_collision_sum_depths_model = []

        conglo_data = pd.read_csv(conglo_path)
        if fixed_n_sampling is None:
            n_sampling = get_next_sampling_id(conglo_data.columns.to_list())
        else:
            n_sampling = fixed_n_sampling
        follow_up_column = f"{column_prefix}{n_sampling}"
        conglo_data[follow_up_column] = False
        conglo_data[follow_up_column + "non_collision"] = ""
        conglo_data[follow_up_column + "contact_sample"] = ""
        conglo_data[follow_up_column + "collision_points"] = ""
        conglo_data[follow_up_column + "collision_sum_depths"] = ""

        grouped = conglo_data.groupby(conglo_data['interaction_type'])

        ##########################################################################
        ######  Shortcut for collision manager calculation
        decimated_envs = {}
        for scene in conglo_data['scene'].unique():
            file_mesh_env, __ = find_files_mesh_env(datasets_dir, scene)
            # decimated_envs[scene] = vedo.vtk2trimesh(vedo.load(file_mesh_env).decimate(fraction=.3))
            decimated_envs[scene] = trimesh.load_mesh(file_mesh_env)
        ##########################################################################

        for current_interaction_type in conglo_data['interaction_type'].unique():

            loss_non_collision_inter_type, loss_contact_inter_type = [], []
            loss_collision_n_points, loss_collision_sum_depths = [], []

            interaction_type_results = grouped.get_group(current_interaction_type)
            if stratified_sampling:
                sample = interaction_type_results.groupby('interaction_type', group_keys=False).apply(lambda x: x.sample(int(np.rint(n_sample_per_interaction_type * len(x) / len(interaction_type_results))))).sample(frac=1)
            else:
                sample = interaction_type_results.sample(n_sample_per_interaction_type)
            sample[follow_up_column + "non_collision"] = 0.0
            sample[follow_up_column + "contact_sample"] = 0.0
            sample[follow_up_column + "collision_points"] = 0.0
            sample[follow_up_column + "collision_sum_depths"] = 0.0
            # sample = conglo_data.sample(n_sample)

            for idx, row in tqdm(sample.iterrows(), total=sample.shape[0]):
                gc.collect()
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
                trimesh_env = decimated_envs[env_name]

                contact_data = measure_trimesh_collision(trimesh_env, trimesh_obj)

                if visualize:
                    s = trimesh.Scene()
                    trimesh_obj.visual.face_colors = [200, 200, 200, 255]
                    s.add_geometry(trimesh_obj)
                    s.add_geometry(trimesh_env)
                    s.show()


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
                sample.loc[idx, [follow_up_column + "non_collision"]] = current_loss_non_coll
                sample.loc[idx, [follow_up_column + "contact_sample"]] = current_loss_contact
                sample.loc[idx, [follow_up_column + "collision_points"]] = current_contact_n_points
                sample.loc[idx, [follow_up_column + "collision_sum_depths"]] = current_contact_sum_depths
                loss_non_collision_inter_type.append(current_loss_non_coll)
                loss_contact_inter_type.append(current_loss_contact)
                loss_collision_n_points.append(current_contact_n_points)
                loss_collision_sum_depths.append(current_contact_sum_depths)

                loss_non_collisions_model.append(current_loss_non_coll)
                loss_contacts_model.append(current_loss_contact)
                loss_collision_n_points_model.append(current_contact_n_points)
                loss_collision_sum_depths_model.append(current_contact_sum_depths)

            # loss_non_collision_env = loss_non_collision_env / n_sample_per_scene
            # loss_contact_env = loss_contact_env / n_sample_per_scene
            # print("   Scene", current_env_name)
            # print('      non_collision score:', loss_non_collision_env)
            # print('      contact score:', loss_contact_env)
            tb_data.append([model, filter_dataset, current_interaction_type,
                            statistics.mean(loss_non_collision_inter_type),
                            statistics.stdev(loss_non_collision_inter_type),
                            statistics.mean(loss_contact_inter_type),
                            statistics.mean(loss_collision_n_points),
                            statistics.stdev(loss_collision_n_points),
                            statistics.mean(loss_collision_sum_depths),
                            statistics.stdev(loss_collision_sum_depths)])

            conglo_data.loc[sample.index.to_list(),[follow_up_column]] =True
            conglo_data.loc[sample.index.to_list(),[follow_up_column + "non_collision"]] = sample.loc[sample.index.to_list(),[follow_up_column + "non_collision"]]
            conglo_data.loc[sample.index.to_list(),[follow_up_column + "contact_sample"]] = sample.loc[sample.index.to_list(),[follow_up_column + "contact_sample"]]
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "collision_points"]] = sample.loc[sample.index.to_list(), [follow_up_column + "collision_points"]]
            conglo_data.loc[sample.index.to_list(), [follow_up_column + "collision_sum_depths"]] = sample.loc[sample.index.to_list(), [follow_up_column + "collision_sum_depths"]]
        # print("  Overall")
        #collision_score =  # sum(loss_non_collisions_model)/len(loss_non_collisions_model)
        #contact_score =  # sum(loss_contacts_model)/len(loss_contacts_model)
        # print('      non_collision score:', collision_score)
        # print('      contact score:', contact_score)
        tb_data.append([model, filter_dataset, "Overall",
                        statistics.mean(loss_non_collisions_model),
                        statistics.stdev(loss_non_collisions_model),
                        statistics.mean(loss_contacts_model),
                        statistics.mean(loss_collision_n_points_model),
                        statistics.stdev(loss_collision_n_points_model),
                        statistics.mean(loss_collision_sum_depths_model),
                        statistics.stdev(loss_collision_sum_depths_model)])

        # print(tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        # print(tabulate(tb_data, headers=tb_headers, floatfmt=".4f", tablefmt="simple"))

        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.INFO, format='')

        logging.info(f"File: {os.path.basename(os.path.realpath(__file__))}")
        logging.info(f"stratified_sampling: {stratified_sampling}")
        logging.info(f"n_sample_per_scene:  {n_sample_per_interaction_type}")
        logging.info(f"filter_dataset: {filter_dataset}")

        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))


        conglo_data.to_csv(conglo_path,index=False)