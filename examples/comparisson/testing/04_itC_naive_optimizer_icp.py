import os
import json
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


    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    directory_datasets = opj(base_dir, "datasets")

    samples_dir = opj(base_dir,'test', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test', 'sampled_it_clearance_opti_icp')

    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    follow_up_column = "it_auto_samples_opti_icp"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not follow_up_column in follow_up_data.columns:
        follow_up_data[follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[follow_up_data[follow_up_column] == False].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[follow_up_data[follow_up_column] == True].index.size

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    for dataset, env_name, interaction in pending_tasks:

        directory_bodies = opj(samples_dir, env_name,interaction)
        it_b0 = trimesh.load(opj(directory_bodies, "body_0.ply"))
        it_b1 = trimesh.load(opj(directory_bodies, "body_1.ply"))
        it_b2 = trimesh.load(opj(directory_bodies, "body_2.ply"))

        file_mesh_env, dataset_name = find_files_mesh_env(directory_datasets, env_name)

        trimesh_decimated_env = vedo.vtk2trimesh(vedo.load(file_mesh_env).decimate(fraction=.3))


        influence_radio_bb = 1.5
        extension, middle_point = it.util.influence_sphere(it_b0, influence_radio_bb)
        tri_mesh_env_cropped = it.util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point, extension)

        s= trimesh.Scene()
        s.add_geometry(it_b0)
        s.add_geometry(tri_mesh_env_cropped)
        s.show()

        matrix, transformation, cost =trimesh.registration.icp(it_b2.vertices, tri_mesh_env_cropped, max_iterations=1)
        print("matrix", matrix)
        # print("transformation", transformation)
        print("cost", cost)
        s = trimesh.Scene()
        transf = trimesh.Trimesh(vertices=transformation, faces=it_b2.faces)
        s.add_geometry(transf)
        s.add_geometry(tri_mesh_env_cropped)
        s.show()


