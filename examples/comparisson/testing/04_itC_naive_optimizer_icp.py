import gc
import os
import json
import random
from os.path import join as opj
from shutil import copyfile

import numpy
import numpy as np
import pandas as pd
import vedo
from vedo import vtk2trimesh
import trimesh

from ctrl.point_selection import ControlPointSelection
from ctrl.sampler import CtrlPropagatorSampler
import it


def find_files_mesh_env(datasets_dir, env_name):
    """
    Return directory were the mesh of an scene is present
    """
    datasets =['prox', "mp3d", "replica_v1"]
    for d in datasets:
        for scene in os.listdir(opj(datasets_dir, d, "scenes")):
            if env_name+".ply" == scene:
                return opj(datasets_dir, d, "scenes", scene), d



if __name__ == '__main__':
    # [ 'reaching_out_mid_up', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_mid',
    # 'sitting_looking_to_right', 'sitting_compact', 'reachin_out_ontable_one_hand'
    # 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot']

    # interaction = 'reaching_out_mid_up'

    visualize= True
    shuffle_order = True
    save_results = False

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    directory_datasets = opj(base_dir, "datasets")

    samples_dir = opj(base_dir,'test', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test', 'sampled_it_clearance_opti_icp')

    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    previus_follow_up_column = "it_auto_samples"
    current_follow_up_column = "it_auto_samples_opti_icp"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[(follow_up_data[current_follow_up_column] == False)
                                         &  (follow_up_data[previus_follow_up_column]==True)].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[follow_up_data[current_follow_up_column] == True].index.size

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    if shuffle_order:
        random.shuffle(pending_tasks)

    last_env_used=None

    for dataset, env_name, interaction in pending_tasks:
        print(dataset, env_name, interaction)
        for i in range(follow_up_data.loc[dataset, env_name, interaction]['num_it_auto_samples']):
            directory_bodies = opj(samples_dir, env_name,interaction)
            it_body = trimesh.load(opj(directory_bodies, f"body_{i}.ply"))

            file_mesh_env = opj(directory_datasets, dataset, "scenes", env_name+".ply")

            if last_env_used != env_name:
                last_env_used = env_name
                trimesh_decimated_env = vedo.vtk2trimesh(vedo.load(file_mesh_env).decimate(fraction=.3))

            influence_radio_bb = 1.5
            extension, middle_point = it.util.influence_sphere(it_body, influence_radio_bb)
            tri_mesh_env_cropped = it.util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point, extension)

            matrix, transformation, cost =trimesh.registration.icp(it_body.vertices, tri_mesh_env_cropped,
                                                                   max_iterations=1, reflection= False,scale= False )
            print("matrix", matrix)
            print("cost", cost)

            if visualize:
                s = trimesh.Scene()
                body_transf = trimesh.Trimesh(vertices=transformation, faces=it_body.faces)
                it_body.visual.face_colors = [200, 200, 200, 150]
                body_transf.visual.face_colors = [200, 200, 200, 255]
                s.add_geometry(it_body)
                s.add_geometry(body_transf)
                s.add_geometry(tri_mesh_env_cropped)
                s.show()

            if save_results:
                output_subdir = opj(output_dir, env_name, interaction)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                it_body.export(opj(output_subdir, f"body_{i}.ply"))

        if save_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset, env_name, interaction), current_follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
        gc.collect()