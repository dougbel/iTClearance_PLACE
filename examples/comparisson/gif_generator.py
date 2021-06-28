import gc
import os
import random
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

def get_files_with_extension(directory, extension):
    return [file_name for file_name in os.listdir(directory) if file_name.endswith(extension)]


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



if __name__ == '__main__':

    visualize = True
    shuffle_order = True
    save_results = False

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    directory_datasets = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test", "sampled_it_clearance")
    samples_it_opti_down_dir = opj(base_dir, "test", "sampled_it_clearance_opti_down_trans")

    samples_place_dir = opj(base_dir, "test", "sampled_place_exec")

    output_dir = opj(base_dir, 'test', 'gifted_place_auto_samples_extracted')

    follow_up_file = opj(base_dir, 'test', 'follow_up_process.csv')
    previus_follow_up_column = "place_auto_samples_extracted"
    current_follow_up_column = "gifted_place_auto_samples_extracted"

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

    if shuffle_order:
        random.shuffle(pending_tasks)

    for dataset_name, env_name, interaction in pending_tasks:

        # if not( env_name == "zsNo4HB9uLZ-bedroom0_0" and interaction == "standing_up"):
        #      continue

        print( dataset_name, env_name, interaction)
        it_subdir = opj(samples_it_dir, env_name, interaction)
        it_opti_down_subdir = opj(samples_it_opti_down_dir, env_name, interaction)
        place_subdir = opj(samples_place_dir, env_name, interaction)

        for np_point_file_name in get_files_with_extension(it_subdir, ".npy"):
            print(np_point_file_name)
            np_point = np.load(opj(it_subdir, np_point_file_name))
            n = np_point_file_name[np_point_file_name.find("_") + 1:np_point_file_name.find(".")]

            file_mesh_env = opj(directory_datasets, dataset_name, "scenes", env_name + ".ply")
            trimesh_env = trimesh.load(file_mesh_env)

            #calculating view center
            view_center = np.copy(np_point)
            if view_center[2] < trimesh_env.vertices.min(axis=0)[2] + .5:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + .5
            elif view_center[2] > trimesh_env.vertices.min(axis=0)[2] + 1:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + 1

            output_subdir = opj(output_dir, env_name, interaction, "place")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_orig.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_orig.gif"))
            trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt1.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opt1.gif"))
            trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt2.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opt2.gif"))

            output_subdir = opj(output_dir, env_name, interaction, "it")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            trimesh_body = trimesh.load(opj(it_subdir, f"body_{n}.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}.gif"))
            trimesh_body = trimesh.load(opj(it_opti_down_subdir, f"body_{n}.ply"))
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{n}_opti_down.gif"))


        if save_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset_name, env_name, interaction), current_follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")

        gc.collect()