"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
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

    marked_as = 4 #simple_random_selection 1207 samples

    filter_dataset = "prox"    # None   prox   mp3d  replica_v1

    visualize=False

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
        follow_up_column = f"{column_prefix}{marked_as}"
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

            sample = interaction_type_results[interaction_type_results[follow_up_column]==True]

            sample[follow_up_column + "collision_points"] = 0.0
            sample[follow_up_column + "collision_sum_depths"] = 0.0
            # sample = conglo_data.sample(n_sample)

            for idx, row in tqdm(sample.iterrows(), total=sample.shape[0]):
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

                current_contact_n_points = len(contact_data)
                current_contact_sum_depths = sum([data.depth for data in contact_data])
                current_loss_non_coll = sample.loc[idx, [follow_up_column + "non_collision"]].values[0]
                current_loss_contact = sample.loc[idx, [follow_up_column + "contact_sample"]].values[0]
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
        logging.info(f"Marked as : {marked_as}")
        logging.info(f"filter_dataset: {filter_dataset}")

        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))

        insert_after = follow_up_column + "contact_sample"
        pos_after = conglo_data.columns.to_list().index(insert_after)
        order_columns = conglo_data.columns.to_list()[:pos_after+1] +conglo_data.columns.to_list()[-2:] +conglo_data.columns.to_list()[pos_after+1:-2]

        conglo_data =conglo_data[order_columns]

        conglo_data.to_csv(conglo_path,index=False)