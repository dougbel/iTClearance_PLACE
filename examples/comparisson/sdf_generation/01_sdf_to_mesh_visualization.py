import json
import os

import numpy as np
import torch
import vedo
from mesh_to_sdf import mesh_to_voxels, scale_to_unit_cube
import trimesh
import time

import skimage.measure as measure

from util.util_mesh import read_full_mesh_sdf
from utils_read_data import define_scene_boundary, read_mesh_sdf

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_dim = 256
    dataset_name = "mp3d"

    scenes_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}/scenes"

    # sdf_dir = f"output/sdf_generation/{dataset_name}/sdf_{grid_dim}"   # visualize from generated sdf
    sdf_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}/sdf"   # visualize from dataset


    for scene_file in os.listdir(scenes_dir): # ['MPH1Library.ply']:

        scene_name = scene_file[:scene_file.find(".ply")]

        mesh_orig = trimesh.load(os.path.join(scenes_dir, scene_file))
        vedo_orig = vedo.load(os.path.join(scenes_dir, scene_file))

        sdf = np.load(os.path.join(sdf_dir, scene_name+"_sdf.npy")).reshape(grid_dim, grid_dim, grid_dim)
        vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0)
        mesh_reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        if os.path.exists(os.path.join(sdf_dir, scene_name+".json")):
            with open(os.path.join(sdf_dir, scene_name+".json")) as f:
                test_data = json.load(f)
            print(f"{scene_name} differences json and mesh")
            print(f"diff MAXs: {test_data['max']-mesh_orig.bounding_box.vertices.max(axis=0)} ")
            print(f"diff MINs: {test_data['min']-mesh_orig.bounding_box.vertices.min(axis=0)} ")
            orig_ext=mesh_orig.bounding_box.extents
            json_ext=test_data['max'] - np.array(test_data['min'])
            print(f"extends orig: {orig_ext} ")
            print(f"extends json: {json_ext} ")
            print(f"proportions: {json_ext[0]/orig_ext[0]}, {json_ext[1]/orig_ext[1]}, {json_ext[2]/orig_ext[2]} ")
            print("---------------------------------------")

        vp = vedo.Plotter(verbose=0, title=scene_name, bg="white", size=(1200, 800), axes=8)
        vedo_orig.backFaceCulling(True)
        vp.add(vedo_orig)
        vp.show()
        vp.close()



        vp = vedo.Plotter(verbose=0,title="Corrected SDF "+scene_name,  bg="white", size=(1200, 800), axes=8)
        vedo_reconstructed = vedo.trimesh2vtk(mesh_reconstructed)
        vedo_reconstructed.c("green")
        vedo_reconstructed.backFaceCulling(True)
        vp.add(vedo_reconstructed)
        vp.add(vedo_orig)
        vp.show()
        vp.close()
