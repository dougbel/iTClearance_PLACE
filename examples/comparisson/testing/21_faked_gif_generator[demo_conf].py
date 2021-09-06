import argparse
import gc
import json
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
parser.add_argument('--batch_size', default=5, help='Dataset name')


opt = parser.parse_args()
print(opt)

if __name__ == '__main__':


    # python examples/comparisson/testing/21_faked_gif_generator.py --base_dir /media/dougbel/Tezcatlipoca/PLACE_trainings --batch_size 5

    register_results = True
    shuffle_order = False
    base_dir = opt.base_dir
    # crop the dataset to an specific size to avoid problem on rendering the gif image
    crop_scene_on_dataset = "replica_v1"  # None

    batch_size = int(opt.batch_size)


    directory_datasets = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test", "sampled_it_clearance")
    samples_it_opti_smplx_dir = opj(base_dir, "test", "sampled_it_clearance_opti_smplx")

    samples_place_dir = opj(base_dir, "test", "sampled_place_exec[demo_conf]")

    output_dir = opj(base_dir, 'test', 'gifted_faked_examples')

    follow_up_file = opj(base_dir, 'test', 'follow_up_process.csv')
    previus_follow_up_column = "gifted_place_auto_samples_extracted[demo_conf]"
    current_follow_up_column = "gifted_faked_examples"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size

    completed_task = list(follow_up_data[(follow_up_data[current_follow_up_column] == True)
                                        & (follow_up_data[previus_follow_up_column] == True)].index)
    num_completed_task = len(completed_task)

    pending_tasks = list(follow_up_data[(follow_up_data[current_follow_up_column] == False)
                                        & (follow_up_data[previus_follow_up_column] == True)].index)
    num_pending_tasks = len(pending_tasks)

    if num_pending_tasks <= 0:
        exit()

    if shuffle_order :
        random.shuffle(pending_tasks)

    print('STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))


    working_on_tasks = pending_tasks[:batch_size]


    for (dataset_name, env_name, interaction) in working_on_tasks:

        print( dataset_name, env_name, interaction)
        it_subdir = opj(samples_it_dir, env_name, interaction)
        it_opti_smplx_subdir = opj(samples_it_opti_smplx_dir, env_name, interaction)
        place_subdir = opj(samples_place_dir, env_name, interaction)

        np_point_files = [f for f in get_file_names_with_extension_in(it_subdir, ".npy") if f.startswith("point_")]
        #select one of the files randomly
        np_point_file_name = np_point_files[ random.randint( 0, len(np_point_files)-1 ) ]

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
            b = vedo.Box(pos=np_point, length=4.6, width=4.6, height=6)
            vedo_env.cutWithMesh(b)
            vedo.write(vedo_env, opj(output_subdir, f"{env_name}_cropped.ply"))
            trimesh_env = trimesh.load(opj(output_subdir, f"{env_name}_cropped.ply"))
            os.remove(opj(output_subdir, f"{env_name}_cropped.ply"))


        z_translation = -0.70
        # if random.uniform(0, 1) < .5:
        #     z_translation = -0.70
        # if random.uniform(0, 1) < .5:
        #     z_translation = 0.70

        #calculating view center with the translation for generate a fake interaction
        view_center = np.copy(np_point)
        if view_center[2] < trimesh_env.vertices.min(axis=0)[2] + .5:
            view_center[2] = trimesh_env.vertices.min(axis=0)[2] + .5 + z_translation
        elif view_center[2] > trimesh_env.vertices.min(axis=0)[2] + 1:
            view_center[2] = trimesh_env.vertices.min(axis=0)[2] + 1 + z_translation

        output_subdir = opj(output_dir, env_name, interaction)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        sample_algorithm=""
        if random.uniform(0,1) <.5 :
            trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt2.ply"))
            sample_algorithm = "place"
        else:
            trimesh_body = trimesh.load(opj(it_opti_smplx_subdir, f"body_{n}.ply"))
            counter_part_algorithm = "it"

        trimesh_body.apply_translation([0, 0, z_translation])
        generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_fake.gif"))

        fake_gif_data = {
            "dataset_name": dataset_name,
            "env_name": env_name,
            "interaction": interaction,
            "sample_algorithm": sample_algorithm,
            "num_point": n
        }
        with open( opj(output_subdir, f"body_fake.json"), 'w') as outfile:
            json.dump(fake_gif_data, outfile, indent=4)

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