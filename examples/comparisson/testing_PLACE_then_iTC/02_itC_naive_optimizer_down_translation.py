import gc
import os
import random
from os.path import join as opj
from shutil import copyfile

import pandas as pd
import trimesh
import vedo

import it
from it import util



if __name__ == '__main__':

    visualize= True
    save_results = False

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson"


    directory_datasets = opj(base_dir, "datasets")


    place_samples_dir = opj(base_dir,'test_place_picker', 'sampled_place_exec')
    it_samples_dir = opj(base_dir, 'test_place_picker', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test_place_picker', 'sampled_it_clearance_opti_down_trans')

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



    follow_up_file = opj(base_dir,'test_place_picker', 'follow_up_process.csv')
    previous_follow_up_column = "num_it_picked_sampled"
    current_follow_up_column = "num_it_picked_sampled_opti_down_trans"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = 0

    comb_dataset_escene = list(follow_up_data[ (follow_up_data[current_follow_up_column] < follow_up_data[previous_follow_up_column] )].index)
    pending_tasks = []
    for dataset_name, scene_name in comb_dataset_escene:
        final = follow_up_data.at[(dataset_name, scene_name), previous_follow_up_column]
        initial = follow_up_data.at[(dataset_name, scene_name), current_follow_up_column]
        for num_point in range(initial, final):
            pending_tasks.append((dataset_name, scene_name, num_point))

    num_pending_tasks = len(pending_tasks)
    num_total_task = follow_up_data['goal_place_picked_sampled'].sum()
    num_completed_task = num_total_task - num_pending_tasks
    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))


    last_env_used=None
    trimesh_decimated_env=None

    for current_dataset_name, current_scene_name, current_num_point in pending_tasks:
        print(current_dataset_name, current_scene_name, current_num_point)

        # extracting information about type of interaction on PLACE execution
        interaction_type_df = pd.read_csv(opj(place_samples_dir, current_scene_name, "interactions.txt"), index_col=0, header=None)
        interaction_type = interaction_type_df.at[current_num_point, 1]
        for current_interaction in interactions_by_type[interaction_type]:
            print(current_dataset_name, current_scene_name, current_num_point, current_interaction)

            directory_bodies = opj(it_samples_dir, current_scene_name)
            it_body_file = opj(directory_bodies, f"body_{current_num_point}_{current_interaction}.ply")
            if(  not os.path.exists(it_body_file) ):
                print(f"WARNING: no {current_interaction} found in point {current_num_point}")
                continue

            it_body = trimesh.load(it_body_file)

            file_mesh_env = opj(directory_datasets, current_dataset_name, "scenes", current_scene_name + ".ply")

            if last_env_used != current_scene_name:
                last_env_used = current_scene_name
                # trimesh_decimated_env = vedo.vtk2trimesh(vedo.load(file_mesh_env).decimate(fraction=.3))
                trimesh_decimated_env = trimesh.load(file_mesh_env)

            influence_radio_bb = 1.5
            extension, middle_point = util.influence_sphere(it_body, influence_radio_bb)
            tri_mesh_env_cropped = util.slide_mesh_by_bounding_box(trimesh_decimated_env, middle_point, extension)


            collision_tester = trimesh.collision.CollisionManager()
            collision_tester.add_object('env', tri_mesh_env_cropped)

            in_collision, contact_data = collision_tester.in_collision_single(it_body, return_data=True)

            translation = 0.0
            delta = -0.003
            while in_collision == False:
                it_body.apply_translation([0, 0, delta])
                translation += delta
                in_collision, contact_data = collision_tester.in_collision_single(it_body, return_data=True)


            if visualize:
                s = trimesh.Scene()
                it_body_orig=trimesh.load(opj(directory_bodies, f"body_{current_num_point}_{current_interaction}.ply"))
                it_body_orig.visual.face_colors = [200, 200, 200, 150]
                it_body.visual.face_colors = [200, 200, 200, 255]
                s.add_geometry(it_body_orig)
                s.add_geometry(it_body)
                s.add_geometry(tri_mesh_env_cropped)
                s.show()

            if save_results:
                output_subdir = opj(output_dir, current_scene_name)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                it_body.export(opj(output_subdir, f"body_{current_num_point}_{current_interaction}.ply"))
                w_txt_file = open(opj(output_subdir, f"body_{current_num_point}_{current_interaction}_translation.txt"), "w")
                w_txt_file.write(str(translation))
                w_txt_file.close()

        if save_results:
            num_completed_task += 1
            num_pending_tasks -= 1
            copyfile(follow_up_file, follow_up_file + "_backup")
            follow_up_data.at[(current_dataset_name, current_scene_name), current_follow_up_column] = current_num_point + 1
            follow_up_data.to_csv(follow_up_file)
            print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
        gc.collect()