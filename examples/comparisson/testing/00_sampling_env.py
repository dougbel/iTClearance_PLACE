import argparse
import json

from it import util
import trimesh
import os
import numpy as np

from it_clearance.utils import calculate_average_distance_nearest_neighbour

if __name__ == "__main__":

    testing_radius = 0.05
    datasets_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets"
    work_directory = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test"

    datasets= ["prox", "mp3d", "replica_v1"]

    for dataset in datasets:
        scenes_dir = os.path.join(datasets_dir, dataset, "scenes")

        for scene_file in os.listdir(scenes_dir):
            scene_name = scene_file[:scene_file.find(".")]
            env_file = os.path.join(scenes_dir, scene_file)
            tri_mesh_env = trimesh.load_mesh(env_file)

            np_sample_points, np_sample_normals = util.sample_points_poisson_disk_radius(tri_mesh_env,
                                                                                         radius=testing_radius)

            output_dir = os.path.join(work_directory, "samples", scene_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            np.save(os.path.join(output_dir, "sample_points.npy"), np_sample_points)
            np.save(os.path.join(output_dir, "sample_normals.npy"), np_sample_normals)

            test_data = {}
            test_data['sampling_radius'] = testing_radius
            test_data['sampled_points'] = np_sample_points.shape[0]
            test_data['sampled_avg_distance'] = calculate_average_distance_nearest_neighbour(np_sample_points)
            with open(os.path.join(output_dir, 'sampling_data.json'), 'w') as outfile:
                json.dump(test_data, outfile, indent=4)


