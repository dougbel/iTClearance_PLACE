import warnings
from os.path import join as opj

import gc
import pandas as pd
import trimesh
from vedo import vtk2trimesh

from ctrl.sampler import CtrlPropagatorSampler
from preprocess.preprocess_optimize import *
from utils import *

warnings.simplefilter("ignore", UserWarning)

def find_files_mesh_env(datasets_dir, env_name):
    """
    Return directory were the mesh of an scene is present
    """
    datasets =['prox', "mp3d", "replica_v1"]
    for d in datasets:
        for scene in os.listdir(opj(datasets_dir, d, "scenes")):
            if env_name+".ply" == scene:
                return opj(datasets_dir, d, "scenes", scene), d

def get_files_with_extension(directory, extension):
    return [file_name for file_name in os.listdir(directory) if file_name.endswith(extension)]


def shift_rotate_mesh(body_verts, body_faces, shift, rotation):
    new_verts = np.zeros(body_verts.shape)
    temp = body_verts - shift
    new_verts[:, 0] = temp[:, 0] * math.cos(-rotation) - temp[:, 1] * math.sin(-rotation)
    new_verts[:, 1] = temp[:, 0] * math.sin(-rotation) + temp[:, 1] * math.cos(-rotation)
    new_verts[:, 2] = temp[:, 2]

    return trimesh.Trimesh(vertices=new_verts, faces=body_faces, face_colors=[200, 200, 200, 255])

def select_it_execution_around_picked_point(data_dir, dataset_name, scene_name, np_point, num_point, interaction_type, output_subdir, visualize = True):

    dataset_path = opj(data_dir, "datasets", dataset_name)
    it_test_results_dir = opj(data_dir, 'test', 'env_test', scene_name)
    directory_json_conf_execution = opj(base_dir, "config", "json_execution")
    directory_descriptors = opj(base_dir, "config", "descriptors_repository")
    directory_of_prop_configs = opj(base_dir, "config", "propagators_configs")


    interactions_by_type={
        "laying": ["laying_bed","laying_hands_up", "laying_on_sofa", "laying_sofa_foot_on_floor"],
        "reaching_out":["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table", "reachin_out_ontable_one_hand"],
        "sitting":["sitting", "sitting_bit_open_arms", "sitting_chair", "sitting_comfortable", "sitting_compact", "sitting_hands_on_device", "sitting_looking_to_right", "sitting_small_table", "sitting_stool", "sitting_stool_one_foot_floor" ],
        "standing_up": ["standing_up", "standup_hand_on_furniture"],
        "walking": ["walking_left_foot", "walking_right_foot"]
    }

    for interaction  in interactions_by_type[interaction_type]:
        # directory_env_test = opj(test_results_dir, scene_name)
        file_mesh_env, dataset_name = find_files_mesh_env(directory_datasets, scene_name)
        file_json_conf_execution = opj(directory_json_conf_execution, f"single_testing_{interaction}.json")
        scores_ctrl = CtrlPropagatorSampler(directory_descriptors, file_json_conf_execution,
                                            it_test_results_dir, directory_of_prop_configs, file_mesh_env)

        N_SAMPLES = 1
        MIN_SCORE = 0.2
        R=1
        for r in [ R/3, R/2, R]:
            vtk_objects, point_samples = scores_ctrl.get_n_samples_around_point(MIN_SCORE, N_SAMPLES, np_point, r,
                                                                        best_in_cluster=True,
                                                                        visualize=visualize)
            if len(vtk_objects) > 0:
                break

        if len(vtk_objects) > 0:
            vtk_object = vtk_objects[0]
            point_sample = point_samples[0]

            np.save(opj(output_subdir, f"point_{num_point}_{interaction}"), point_sample)
            vtk2trimesh(vtk_object).export(opj(output_subdir, f"body_{num_point}_{interaction}.ply"))

            if visualize:
                # read scene mesh
                scene_trimesh = trimesh.load(os.path.join(dataset_path, "scenes", scene_name + '.ply'))
                principal_point = trimesh.primitives.Sphere(radius=0.1, center=np_point)
                principal_point.visual.face_colors = [0, 0, 255]
                transform = np.eye(4)
                transform[:3, 3] = np_point
                cylinder = trimesh.primitives.Cylinder(height=4, radius=1, transform=transform)
                cylinder.visual.face_colors = [100, 255, 100, 100]
                s = trimesh.Scene()
                s.add_geometry(scene_trimesh)
                s.add_geometry(principal_point)
                s.add_geometry(vtk2trimesh(vtk_object))
                s.add_geometry(cylinder)
                s.show(caption=scene_name)



if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson"

    visualize= False

    directory_datasets = opj(base_dir, "datasets")

    follow_up_file = opj(base_dir,'test_place_picker', 'follow_up_process.csv')
    output_dir = opj(base_dir, 'test_place_picker', 'sampled_it_clearance')
    points_dir = opj(base_dir, 'test_place_picker', 'sampled_place_exec')


    current_follow_up_column = "num_it_picked_sampled"
    previus_follow_up_column = "num_place_picked_sampled"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = 0

    comb_dataset_escene = list(follow_up_data[ (follow_up_data[current_follow_up_column] < follow_up_data[previus_follow_up_column] )].index)
    pending_tasks=[]
    for dataset_name, scene_name in comb_dataset_escene:
        final = follow_up_data.at[(dataset_name, scene_name), previus_follow_up_column]
        initial = follow_up_data.at[(dataset_name, scene_name), current_follow_up_column]
        for i in range(initial, final):
            pending_tasks.append((dataset_name, scene_name, i))


    num_pending_tasks = len(pending_tasks)

    num_total_task = follow_up_data['goal_place_picked_sampled'].sum()

    num_completed_task = num_total_task - num_pending_tasks

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    for dataset_name, scene_name, num_point in pending_tasks:
        points_subdir = opj(points_dir, scene_name)
        output_subdir = opj(output_dir, scene_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        np_point = np.load(opj(points_subdir, f"point_{num_point}.npy"))
        interaction_type_df = pd.read_csv(opj(points_subdir, "interactions.txt"), index_col=0, header=None)
        interaction_type = interaction_type_df.at[num_point,1]
        select_it_execution_around_picked_point(base_dir, dataset_name, scene_name, np_point, num_point, interaction_type, output_subdir, visualize=visualize)

        num_completed_task += 1
        num_pending_tasks -= 1
        follow_up_data.at[(dataset_name, scene_name), current_follow_up_column] = (num_point+1)
        follow_up_data.to_csv(follow_up_file)
        print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
        gc.collect()