import argparse
import gc
import os
import random
import shutil
import time

import numpy as np
from shutil import copyfile

import pandas as pd
from  os.path import join as opj

import vedo
import trimesh
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

from util.utils_files import get_file_names_with_extension_in


def generate_gif(trimesh_env, trimesh_body, view_center,  save_on_file=None):

    record=True if save_on_file is not None else False


    env_mesh = Mesh.from_trimesh(trimesh_env)

    trimesh_body.visual.face_colors = [10, 220, 10, 255]
    body_mesh = Mesh.from_trimesh(trimesh_body, smooth=False)

    trimesh_body.visual.face_colors =  [40, 40, 40, 150]
    body_mesh_w = Mesh.from_trimesh(trimesh_body, smooth=False, wireframe=True)

    # reference_mesh = Mesh.from_trimesh(trimesh.primitives.Sphere(center = np_point), smooth=False, wireframe=True)

    scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5]))
    # light = DirectionalLight(color=[.80, .80, .80], intensity=1.0)
    scene.add(env_mesh)
    scene.add(body_mesh)
    scene.add(body_mesh_w)
    # scene.add(reference_mesh)
    # scene.add(light)
    v = Viewer(scene,
               run_in_thread=True,
               render_flags={"shadows": True},
               viewer_flags={"rotate": True,
                             "rotate_axis": [0, 0, 1],
                             "view_center": view_center,
                             "rotate_rate": np.pi / 4.0,
                             "record": record,
                             "use_perspective_cam": True
                             })

    # for "use_perspective_cam": True
    v._trackball.scroll(10)

    # for "use_perspective_cam": False
    # dy=.25
    # spfc = 0.95
    # sf = 1.0
    # sf = spfc * dy
    # c = v._camera_node.camera
    # v._camera_node.camera.xmag = max(c.xmag * sf, 1e-8)
    # v._camera_node.camera.ymag = max(c.ymag * sf, 1e-8 * c.ymag / c.xmag)

    v._trackball.rotate(np.pi / 5, [0, 1, 0])

    time.sleep(8.1)
    v.close_external()
    while v.is_active:
        pass
    if record:
        v.save_gif(save_on_file)


parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--base_dir', required=True, help='Information directory (dataset, pretrained models, etc)')
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--env_name', required=True, help='Environment name')
parser.add_argument('--interaction', required=True, help='Interaction name')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    # python examples/comparisson/testing/20_gif_generator[demo_config].py --base_dir /media/dougbel/Tezcatlipoca/PLACE_trainings --dataset prox --env_name MPH16 --interaction sitting_compact
    # python examples/comparisson/testing/20_gif_generator[demo_config].py --base_dir /media/dougbel/Tezcatlipoca/PLACE_trainings --dataset prox --env_name MPH16 --interaction sitting_hands_on_device

    register_results = True

    # crop the dataset to an specific size to avoid problem on rendering the gif image
    crop_scene_on_dataset = "replica_v1" # None

    base_dir = opt.base_dir
    dataset_name = opt.dataset
    env_name = opt.env_name
    interaction = opt.interaction


    directory_datasets = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test", "sampled_it_clearance")
    # samples_it_opti_down_dir = opj(base_dir, "test", "sampled_it_clearance_opti_down_trans")
    samples_it_opti_smplx_dir = opj(base_dir, "test", "sampled_it_clearance_opti_smplx")

    samples_place_dir = opj(base_dir, "test", "sampled_place_exec[demo_conf]")

    #this is the parh to verify if the gif asociates with iTClearance->SMPLX optimization already exists
    paper_output_dir = opj(base_dir, 'test', 'gifted_place_auto_samples_extracted')

    #this is the actual output directory
    output_dir = opj(base_dir, 'test', 'gifted_place_auto_samples_extracted[demo_conf]')

    follow_up_file = opj(base_dir, 'test', 'follow_up_process.csv')
    previus_follow_up_column = "place_auto_samples_extracted[demo_conf]"
    current_follow_up_column = "gifted_place_auto_samples_extracted[demo_conf]"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[(follow_up_data[current_follow_up_column] == False)
                                        & (follow_up_data[previus_follow_up_column] == True)].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[(follow_up_data[current_follow_up_column] == True)
                                        & (follow_up_data[previus_follow_up_column] == True)].index.size

    print('STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))


    if (dataset_name, env_name, interaction) not in pending_tasks:
        print(f"PREVIOUSLY PERFORMED TASK NOTHING TO DO: base_dir {base_dir}, dataset {dataset_name}, env_name {env_name}, interaction {interaction}")
    else:
        print( dataset_name, env_name, interaction)
        it_subdir = opj(samples_it_dir, env_name, interaction)
        # it_opti_down_subdir = opj(samples_it_opti_down_dir, env_name, interaction)
        it_opti_smplx_subdir = opj(samples_it_opti_smplx_dir, env_name, interaction)

        place_subdir = opj(samples_place_dir, env_name, interaction)

        for np_point_file_name in get_file_names_with_extension_in(it_subdir, ".npy"):
            if not np_point_file_name.startswith("point_"):
                continue
            print(np_point_file_name)
            np_point = np.load(opj(it_subdir, np_point_file_name))
            n = np_point_file_name[np_point_file_name.find("_") + 1:np_point_file_name.find(".")]

            file_mesh_env = opj(directory_datasets, dataset_name, "scenes", env_name + ".ply")
            trimesh_env = trimesh.load(file_mesh_env)

            if crop_scene_on_dataset == dataset_name:
                output_subdir = opj(output_dir, env_name, interaction)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                vedo_env = vedo.load(file_mesh_env)
                b=vedo.Box(pos= np_point,length=4.6, width=4.6, height=6)
                vedo_env.cutWithMesh(b)
                vedo.write(vedo_env, opj(output_subdir, f"{env_name}_cropped.ply"))
                trimesh_env = trimesh.load(opj(output_subdir, f"{env_name}_cropped.ply"))
                os.remove(opj(output_subdir, f"{env_name}_cropped.ply"))


            #calculating view center
            view_center = np.copy(np_point)
            if view_center[2] < trimesh_env.vertices.min(axis=0)[2] + .5:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + .5
            elif view_center[2] > trimesh_env.vertices.min(axis=0)[2] + 1:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + 1

            output_subdir = opj(output_dir, env_name, interaction, "place")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_orig.ply"))
            # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_orig.gif"))
            # gc.collect()
            # trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt1.ply"))
            # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opt1.gif"))
            # gc.collect()
            trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt2.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opt2.gif"))
            gc.collect()



            output_subdir = opj(output_dir, env_name, interaction, "it")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # here I verify if the gif whit iT Clearance with SMPLX optimization is vailible, then copy
            paper_iT_gif_file = opj( paper_output_dir, env_name, interaction, "it", f"body_{n}_opti_smplx.gif")
            if os.path.exists(paper_iT_gif_file):
                shutil.copy(paper_iT_gif_file, opj(output_subdir, f"body_{n}_opti_smplx.gif"))
            else:

                # trimesh_body = trimesh.load(opj(it_subdir, f"body_{n}.ply"))
                # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}.gif"))
                # gc.collect()
                # trimesh_body = trimesh.load(opj(it_opti_down_subdir, f"body_{n}.ply"))
                # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opti_smplx.gif"))
                # gc.collect()
                trimesh_body = trimesh.load(opj(it_opti_smplx_subdir, f"body_{n}.ply"))
                generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opti_smplx.gif"))
                gc.collect()



        if register_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset_name, env_name, interaction), current_follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")

        gc.collect()
        print(f"TASK DONE: base_dir {base_dir}, dataset {dataset_name}, env_name {env_name}, interaction {interaction}")