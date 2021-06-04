"""
Create all json configurations files for the descriptors in folder
"""

import json
import logging
import os
import time
from os.path import join as opj

import open3d as o3d
import trimesh

import it.util as util
from it_clearance.testing.envirotester import EnviroTesterClearance

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('Reading configuration interactions to test')

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    descriptors_dir = opj(base_dir, "config", "descriptors_repository")
    testing_conf_dir= opj(base_dir, "config", "json_execution")

    # trainings_to_test= os.listdir(descriptors_dir)
    # trainings_to_test = ["walking_right_foot"]
    # trainings_to_test = ["sitting_looking_to_right"]
    # trainings_to_test = ["sitting_stool_one_foot_floor"]
    # trainings_to_test = ["sitting"]
    # trainings_to_test = ["sitting_bit_open_arms"]
    # trainings_to_test = ["sitting_chair"]
    # trainings_to_test = ["sitting_small_table"]
    # trainings_to_test = ["laying_hands_up"]
    # trainings_to_test = ["laying_on_sofa"]
    # trainings_to_test = ["standing_up"]
    # trainings_to_test = ["laying_sofa_foot_on_floor"]
    # trainings_to_test = ["standup_hand_on_furniture"]
    trainings_to_test = ["walking_right_foot"]

    base_output_dir = 'output/testing_env_single/'


    json_training_file = None
    for descriptor in trainings_to_test:
        sub_dir = os.path.join(descriptors_dir, descriptor)
        for file_name in os.listdir(sub_dir):
            if ".json" in file_name:
                json_training_file = opj(sub_dir, file_name)

        with open(json_training_file) as f:
            train_data = json.load(f)

        json_conf_execution_file= opj(testing_conf_dir, f"single_testing_{train_data['affordance_name']}.json")
        with open(json_conf_execution_file) as f:
            test_conf_data = json.load(f)

        scene_name = train_data['env_name'][:train_data['env_name'].index("_")]

        # on artificial living room 1
        env_file = opj(base_dir, "datasets", "prox", "scenes", scene_name + ".ply")
        env_file_filled = opj(base_dir, "datasets", "prox", "scenes_filled", scene_name + ".ply")



        output_dir = os.path.join(base_output_dir, scene_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        testing_radius = 0.05

        ####################################
        # 8 orientations
        directory_of_trainings = opj(base_dir, "config", "descriptors_repository") #"./output/descriptors_repository"


        tri_mesh_env = trimesh.load_mesh(env_file)
        tri_mesh_env_filled = trimesh.load_mesh(env_file_filled)

        start = time.time()  # timing execution
        np_test_points, np_env_normals = util.sample_points_poisson_disk_radius(tri_mesh_env, radius=testing_radius)
        end = time.time()  # timing execution
        print("Sampling 1 Execution time: ", end - start)

        # start = time.time()  # timing execution
        # sampling_size = np_test_points.shape[0]
        # np_test_points = util.sample_points_poisson_disk(tri_mesh_env, sampling_size)
        # np_env_normals = util.get_normal_nearest_point_in_mesh(tri_mesh_env, np_test_points)
        # end = time.time()  # timing execution
        # print("Sampling 2 Execution time: ", end - start)

        tester = EnviroTesterClearance(directory_of_trainings, json_conf_execution_file)

        affordance_name = tester.affordances[0][0]
        affordance_object = tester.affordances[0][1]
        tri_mesh_object_file = tester.objs_filenames[0]

        tri_mesh_obj = trimesh.load_mesh(tri_mesh_object_file)

        start = time.time()  # timing execution
        # Testing iT
        full_data_frame = tester.start_full_test(tri_mesh_env, tri_mesh_env_filled, np_test_points, np_env_normals)
        end = time.time()  # timing execution
        time_exe = end - start
        print("Testing execution time: ", time_exe)

        # ##################################################################################################################
        # SAVING output

        output_dir = opj(output_dir, affordance_name + '_' + affordance_object)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info('Saving results on ' + output_dir)

        data = {'execution_time_it_test': time_exe,
                'num_points_tested': np_test_points.shape[0],
                'testing_radius': testing_radius,
                'tester_info': tester.configuration_data,
                'directory_of_trainings': directory_of_trainings,
                'file_env': env_file
                }

        with open(opj(output_dir, 'test_data.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

        tri_mesh_env.export(output_dir + "/test_environment.ply", "ply")
        tri_mesh_obj.export(output_dir + "/test_object.ply", "ply")

        # test points
        o3d_test_points = o3d.geometry.PointCloud()
        o3d_test_points.points = o3d.utility.Vector3dVector(np_test_points)
        o3d.io.write_point_cloud(output_dir + "/test_tested_points.pcd", o3d_test_points)

        # it test
        filename = opj(output_dir, "test_scores.csv")  # "%s/test_scores.csv" % output_dir
        full_data_frame.to_csv(filename)
