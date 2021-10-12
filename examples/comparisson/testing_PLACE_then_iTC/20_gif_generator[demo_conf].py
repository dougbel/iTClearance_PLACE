import argparse
import gc
import os
import time
import imageio

from os.path import join as opj
from shutil import copyfile

import numpy as np
import pandas as pd
import trimesh
import vedo
from pyrender import Mesh, Scene, Viewer, MetallicRoughnessMaterial


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
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--env_name', required=True, help='Environment name')
parser.add_argument('--num_point', required=True, help='num_point name')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    # python examples/comparisson/testing_PLAqCE_then_iTC/20_gif_generator.py --base_dir /media/dougbel/Tezcatlipoca/PLACE_trainings --dataset prox --env_name MPH16 --num_point 0
    # python examples/comparisson/testing_PLACE_then_iTC/20_gif_generator.py --base_dir /media/apacheco/Ehecatl/PLACE_comparisson --dataset prox --env_name MPH16 --num_point 1

    register_results = True
    # crop the dataset to an specific size to avoid problem on rendering the gif image
    crop_scene_on_dataset = "all"  # None

    base_dir = opt.base_dir
    dataset_name = opt.dataset
    env_name = opt.env_name
    num_point = int(opt.num_point)


    directory_datasets = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_it_clearance")
    sampled_it_opti_smplx_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_it_clearance_opti_smplx")
    samples_place_dir = opj(base_dir, "test_place_picker[demo_conf]", "sampled_place_exec")
    points_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'sampled_place_exec')


    output_dir = opj(base_dir, 'test_place_picker[demo_conf]', 'gifted_samples_extracted')

    interactions_by_type = {
        "laying": ["laying_bed", "laying_hands_up", "laying_on_sofa", "laying_sofa_foot_on_floor"],
        "reaching_out": ["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table",
                         "reaching_out_ontable_one_hand"],
        "sitting": ["sitting", "sitting_bit_open_arms", "sitting_chair", "sitting_comfortable", "sitting_compact",
                    "sitting_hands_on_device", "sitting_looking_to_right", "sitting_small_table", "sitting_stool",
                    "sitting_stool_one_foot_floor"],
        "standing_up": ["standing_up", "standup_hand_on_furniture"],
        "walking": ["walking_left_foot", "walking_right_foot"]
    }


    follow_up_file = opj(base_dir, 'test_place_picker[demo_conf]', 'follow_up_process.csv')
    last_task_colum = "num_it_picked_sampled_opti_smplx"
    current_follow_up_column = "gifted_place_auto_samples_extracted"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = 0

    completed_task = []
    for dataset, scene in follow_up_data.index:
        final = follow_up_data.at[(dataset, scene), current_follow_up_column]
        for n in range(0, final):
            completed_task.append((dataset, scene, n))

    num_completed_task = len(completed_task)
    num_total_task = follow_up_data['goal_place_picked_sampled'].sum()

    print('STARTING TASKS: total %d, done %d' % (num_total_task, num_completed_task))


    if (dataset_name, env_name, num_point) in completed_task:
        print(f"PREVIOUSLY PERFORMED TASK NOTHING TO DO: base_dir {base_dir}, dataset {dataset_name}, env_name {env_name}, num_point {num_point}")
    else:
        print( dataset_name, env_name, num_point)
        it_subdir = opj(samples_it_dir, env_name)
        it_opti_smplx_subdir = opj(sampled_it_opti_smplx_dir, env_name)
        place_subdir = opj(samples_place_dir, env_name)

        np_point = np.load(opj(points_dir, env_name, f"point_{num_point}.npy"))

        file_mesh_env = opj(directory_datasets, dataset_name, "scenes", env_name + ".ply")
        trimesh_env = trimesh.load(file_mesh_env)

        if crop_scene_on_dataset == dataset_name or crop_scene_on_dataset=="all":
            print("cropping ", dataset_name)
            output_subdir = opj(output_dir, env_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            vedo_env = vedo.load(file_mesh_env)
            b = vedo.Box(pos=np_point, length=4.6, width=4.6, height=6)
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

        output_subdir = opj(output_dir, env_name, "place")
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # trimesh_body = trimesh.load(opj(place_subdir, f"body_{num_point}_orig.ply"))
        # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{num_point}_orig.gif"))
        # gc.collect()
        # trimesh_body = trimesh.load(opj(place_subdir, f"body_{num_point}_opt1.ply"))
        # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{num_point}_opt1.gif"))
        # gc.collect()
        trimesh_body = trimesh.load(opj(place_subdir, f"body_{num_point}_opt2.ply"))
        generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{num_point}_opt2.gif"))
        gc.collect()


        output_subdir = opj(output_dir, env_name, "it")
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        interaction_type_df = pd.read_csv(opj(samples_place_dir, env_name, "interactions.txt"), index_col=0,header=None)
        interaction_type = interaction_type_df.at[num_point, 1]

        for current_interaction in interactions_by_type[interaction_type]:
            file_name = f"body_{num_point}_{current_interaction}.ply"
            if os.path.exists(opj(samples_it_dir, env_name, file_name)):
                # trimesh_body = trimesh.load(opj(it_subdir, f"body_{num_point}_{current_interaction}.ply"))
                # generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{num_point}_{current_interaction}.gif"))
                # gc.collect()
                trimesh_body = trimesh.load(opj(it_opti_smplx_subdir, f"body_{num_point}_{current_interaction}.ply"))
                generate_gif(trimesh_env, trimesh_body, view_center, opj(output_subdir, f"body_{num_point}_{current_interaction}_opti_smplx.gif"))
                gc.collect()


        if register_results:
            num_completed_task += 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            prev_val = follow_up_data.at[(dataset_name, env_name), current_follow_up_column]
            follow_up_data.at[(dataset_name, env_name), current_follow_up_column] = prev_val+1
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}")

        gc.collect()
        print(f"TASK DONE: base_dir {base_dir}, dataset {dataset_name}, env_name {env_name}, num_point {num_point}")