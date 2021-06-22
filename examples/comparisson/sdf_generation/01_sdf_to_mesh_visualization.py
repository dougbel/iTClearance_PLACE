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
    dataset_name = "replica_v1"
    base_dir = f"output/sdf_generation/{dataset_name}/sdf_{grid_dim}"


    scenes_dir = os.path.join(base_dir, dataset_name,  "scenes")
    sdf_dir = os.path.join(base_dir, dataset_name, "sdf")


    for scene_file in os.listdir(scenes_dir): # ['MPH1Library.ply']:

        scene_name = scene_file[:scene_file.find(".ply")]

        mesh_orig = trimesh.load(os.path.join(scenes_dir, scene_file))

        sdf = np.load(os.path.join(sdf_dir, scene_name+"_sdf.npy")).reshape(grid_dim, grid_dim, grid_dim)
        vertices, faces, normals, _ = measure.marching_cubes(sdf, level=0)
        mesh_reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        vp = vedo.Plotter(verbose=0,title=scene_name,  bg="white", size=(1200, 800), axes=8)
        vedo_reconstructed = vedo.trimesh2vtk(mesh_reconstructed)
        vedo_reconstructed.c("green")
        vedo_reconstructed.backFaceCulling(True)
        vp.add(vedo_reconstructed)
        vp.add(vedo.trimesh2vtk(mesh_orig))
        vp.show()
        vp.close()
