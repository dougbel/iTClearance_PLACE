"""
This code is used to generate sdf files using the approach in https://github.com/marian42/mesh_to_sdf
"""

import os

import numpy as np
from mesh_to_sdf import mesh_to_voxels, scale_to_unit_cube
import trimesh
import time


if __name__ == "__main__":

    dataset_name = "prox"
    base_dir = f"/media/apacheco/Ehecatl/PLACE_comparisson/datasets/{dataset_name}"
    scenes_dir = os.path.join(base_dir, "scenes")
    grid_dim= 1024
    output_dir = f"output/sdf_generation/{dataset_name}/sdf_{grid_dim}"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for scene_file in os.listdir(scenes_dir):
        print("working on "+scene_file)
        scene_name = scene_file[:scene_file.find(".ply")]

        # mesh_orig = trimesh.load('chair.obj')
        mesh_orig = trimesh.load(os.path.join(scenes_dir, scene_file))

        mesh_scaled = scale_to_unit_cube(mesh_orig)
        mesh_scaled.export(os.path.join(output_dir, scene_name+"_scaled_to_unit_cube.ply"))

        start = time.time()

        voxels = mesh_to_voxels(mesh_orig, grid_dim, sign_method='normal', pad=False)

        end = time.time()
        execution_time = end - start
        print(f"sdf calculation on {scene_file} execution time {execution_time} [s]")

        np.save(os.path.join(output_dir, scene_name+"_sdf.npy"), voxels)
