"""
This program is for seen what they mention as a swap between axis Z and Y

    IT SEEMS IT IS ONLY FOR THE sdf files
"""

import os
import numpy as np

import pandas as pd
from os.path import join as opj
from trimesh.base import Trimesh

import vedo
import trimesh

if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    mp3d_path = opj(base_dir, "datasets_raw", "mp3", "scenes")

    for env in os.listdir(mp3d_path):
        trimesh_scene = trimesh.load(opj(mp3d_path, env))

        cur_scene_verts = np.zeros(np.asarray(trimesh_scene.vertices).shape)
        cur_scene_verts[:, 0] = np.asarray(trimesh_scene.vertices)[:, 0]
        cur_scene_verts[:, 1] = np.asarray(trimesh_scene.vertices)[:, 2]
        cur_scene_verts[:, 2] = np.asarray(trimesh_scene.vertices)[:, 1]

        trimesh_scene_swap = Trimesh(vertices=cur_scene_verts, faces=trimesh_scene.faces)

        plt = vedo.Plotter(N=2, size=(1024 * 2, 768), axes=1)
        plt.show(trimesh_scene, at=0)
        plt.show(trimesh_scene_swap, at=1, interactive=1).close()

