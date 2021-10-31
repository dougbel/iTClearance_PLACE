import warnings
from os.path import join as opj
from shutil import copyfile
import gc

import numpy as np
import pandas as pd
import trimesh
import vedo

from tqdm import tqdm
import torch.optim as optim

from util.util_interactive import Selector
from util.util_mesh import read_full_mesh_sdf, define_scene_boundary_on_the_fly
from util.util_preprocessing import crop_scene_cube_smplx_at_point

from human_body_prior.tools.model_loader import load_vposer

import smplx
import chamfer_pytorch.dist_chamfer as ext
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from util.utils_files import get_file_names_with_extension_in
from utils import *

warnings.simplefilter("ignore", UserWarning)



def create_marker(np_point):
    transform = np.eye(4)
    transform[:3, 3] = np_point
    marker = trimesh.primitives.Cylinder(height=0.02, radius=0.05, transform=transform)
    marker += trimesh.primitives.Cylinder(height=0.1, radius=0.02, transform=transform)
    marker.visual.face_colors = [0, 255, 0, 255]
    return marker

if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    directory_datasets = opj(base_dir, "datasets")

    data_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'sampled_place_exec')

    follow_up_file = opj(base_dir,'test_place_picker[demo_conf]', 'follow_up_process.csv')
    counter_follow_up_column = "num_place_picked_sampled"
    goal_follow_up_column = "goal_place_picked_sampled"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])


    data_set_scene = list(follow_up_data.index)

    random.shuffle(data_set_scene)

    for dataset_name, scene_name in data_set_scene:
        data_subdir = opj(data_dir, scene_name)
        print(dataset_name+"_"+scene_name)

        trimesh_env = trimesh.load(opj(directory_datasets, dataset_name, "scenes", f"{scene_name}.ply"))

        for np_point_file_name in get_file_names_with_extension_in(data_subdir, ".npy"):
            np_point =  np.load(opj(data_subdir, np_point_file_name))
            trimesh_env += create_marker(np_point)

        num_place_picked_sampled = follow_up_data.at[(dataset_name, scene_name), "num_place_picked_sampled"]
        goal_place_picked_sampled = follow_up_data.at[(dataset_name, scene_name), "goal_place_picked_sampled"]


        trimesh_env.visual.face_colors = trimesh_env.visual.face_colors

        vedo_env = vedo.utils.trimesh2vtk(trimesh_env)
        vedo_env.backFaceCulling(True)
        vedo_env.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp = vedo.Plotter(bg="white", size=(1200, 800), axes=0)
        vp.show(vedo_env)
        vp.close()
