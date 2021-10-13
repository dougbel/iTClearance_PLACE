import argparse
import gc
import json
import os
import random
import time

import imageio
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

    material = MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.25, 0.7, 0.25, 1.0))

    # trimesh_body.visual.face_colors = [10, 220, 10, 255]
    body_mesh = Mesh.from_trimesh(trimesh_body, smooth=True, material=material)

    # trimesh_body.visual.face_colors =  [40, 40, 40, 150]
    # body_mesh_w = Mesh.from_trimesh(trimesh_body, smooth=False, wireframe=True)

    # reference_mesh = Mesh.from_trimesh(trimesh.primitives.Sphere(center = np_point), smooth=False, wireframe=True)

    scene = Scene(ambient_light=np.array([0.1, 0.1, 0.1]))
    # light = DirectionalLight(color=[.80, .80, .80], intensity=1.0)
    scene.add(env_mesh)
    scene.add(body_mesh)
    # scene.add(body_mesh_w)
    # scene.add(reference_mesh)
    # scene.add(light)
    v = Viewer(scene,
               run_in_thread=True,
               viewport_size=(420, 315),
               render_flags={"shadows": False},
               viewer_flags={"rotate": True,
                             "rotate_axis": [0, 0, 1],
                             "view_center": view_center,
                             "rotate_rate": np.pi / 4.0,
                             "record": record,
                             "use_perspective_cam": True,
                             "refresh_rate": 24,
                             "use_raymond_lighting": True
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

    v._trackball.rotate(np.pi / 6.4, [0, 1, 0])

    time.sleep(8.1)
    v.close_external()
    while v.is_active:
        pass
    if record:
        # v.save_gif(save_on_file)
        imageio.mimwrite(save_on_file,  v._saved_frames, fps=v.viewer_flags['refresh_rate'],
                         palettesize=256, subrectangles=True)
        v._saved_frames = []


parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--base_dir', required=True, help='Information directory (dataset, pretrained models, etc)')
parser.add_argument('--batch_size', default=5, help='Dataset name')


opt = parser.parse_args()
print(opt)

if __name__ == '__main__':


    # python examples/comparisson/testing_PLACE_then_iTC/21_faked_gif_generator-PLACE_then_iTC[demo_conf].py --base_dir /media/dougbel/Tezcatlipoca/PLACE_trainings --batch_size 2

    register_results = True
    shuffle_order = False
    num_fakes_per_env = 3
    base_dir = opt.base_dir
    # crop the dataset to an specific size to avoid problem on rendering the gif image
    crop_scene_on_dataset = "all"  # None

    batch_size = int(opt.batch_size)


    directory_datasets = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_it_clearance")
    samples_it_opti_smplx_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_it_clearance_opti_smplx")

    samples_place_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_place_exec")

    output_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'gifted_faked_examples')

    follow_up_file = opj(base_dir, 'test_place_picker[demo_conf]', 'follow_up_process.csv')
    num_available_samples_column =  "num_it_picked_sampled_opti_smplx"
    current_follow_up_column = "gifted_faked_examples"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False


    completed_task = []
    pending_tasks =[]
    for dataset, scene in follow_up_data.index:
        is_done = follow_up_data.at[(dataset, scene), current_follow_up_column]
        if is_done :
            completed_task.append((dataset, scene))
        else:
            pending_tasks.append((dataset, scene))

    num_completed_task = len(completed_task)
    num_pending_tasks = len(pending_tasks)
    num_total_task = follow_up_data.index.size

    if num_pending_tasks <= 0:
        exit()

    if shuffle_order :
        random.shuffle(pending_tasks)

    print('STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))


    working_on_tasks = pending_tasks[:batch_size]


    for (dataset_name, env_name) in working_on_tasks:

        print( dataset_name, env_name)

        available_samples= follow_up_data.at[(dataset_name, env_name), num_available_samples_column]
        points_selection = np.random.choice(range(available_samples), num_fakes_per_env, replace=False)
        points_selection.sort()
        it_subdir = opj(samples_it_dir, env_name)
        it_opti_smplx_subdir = opj(samples_it_opti_smplx_dir, env_name)
        place_subdir = opj(samples_place_dir, env_name)

        for n_point in points_selection:

            np_point_files = [f for f in get_file_names_with_extension_in(it_subdir, ".npy") if f.startswith(f"point_{n_point}")]
            if  len(np_point_files) ==0:
                print("No points information for ", dataset_name, env_name)
                continue

            #select one of the files randomly
            n_interaction_selected = random.randint(0, len(np_point_files) - 1)
            np_point_file_name_selected = np_point_files[n_interaction_selected]
            interaction_selected = np_point_file_name_selected.replace(f"point_{n_point}_","").replace(".npy", "")
            print(np_point_file_name_selected)

            # PLACE position of the sample is used cause  ITClearance gif used this same one
            np_point_selected = np.load(opj(place_subdir, f"point_{n_point}.npy"))

            file_mesh_env = opj(directory_datasets, dataset_name, "scenes", env_name + ".ply")
            trimesh_env = trimesh.load(file_mesh_env)

            if crop_scene_on_dataset == dataset_name or crop_scene_on_dataset=="all":
                print("cropping ", dataset_name)
                output_subdir = opj(output_dir, env_name)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                vedo_env = vedo.load(file_mesh_env)
                b = vedo.Box(pos=np_point_selected, length=4.6, width=4.6, height=6)
                vedo_env.cutWithMesh(b)
                vedo.write(vedo_env, opj(output_subdir, f"{env_name}_cropped.ply"))
                trimesh_env = trimesh.load(opj(output_subdir, f"{env_name}_cropped.ply"))
                os.remove(opj(output_subdir, f"{env_name}_cropped.ply"))


            z_translation = -0.60

            #calculating view center with the translation for generate a fake interaction
            view_center = np.copy(np_point_selected)
            if view_center[2] < trimesh_env.vertices.min(axis=0)[2] + .5:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + .5 + z_translation
            elif view_center[2] > trimesh_env.vertices.min(axis=0)[2] + 1:
                view_center[2] = trimesh_env.vertices.min(axis=0)[2] + 1 + z_translation

            output_subdir = opj(output_dir, env_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            sample_algorithm=""
            # if random.uniform(0,1) <.5 :
            #     trimesh_body = trimesh.load(opj(place_subdir, f"body_{n}_opt2.ply"))
            #     sample_algorithm = "place"
            # else:
            trimesh_body = trimesh.load(opj(it_opti_smplx_subdir, f"body_{n_point}_{interaction_selected}.ply"))
            sample_algorithm = "it"

            trimesh_body.apply_translation([0, 0, z_translation])
            generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_fake_{n_point}_{interaction_selected}.gif"))

            fake_gif_data = {
                "dataset_name": dataset_name,
                "env_name": env_name,
                "interaction": interaction_selected,
                "sample_algorithm": sample_algorithm,
                "num_point": str(n_point)
            }
            with open(opj(output_subdir, f"body_fake_{n_point}_{interaction_selected}.json"), 'w') as outfile:
                json.dump(fake_gif_data, outfile, indent=4)

            gc.collect()

        if register_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(dataset_name, env_name), current_follow_up_column] = True
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pending {num_pending_tasks}")

        gc.collect()
        print(f"TASK DONE: base_dir {base_dir}, dataset {dataset_name}, env_name {env_name}")