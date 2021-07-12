import os
import random

import pandas as pd
from  os.path import join as opj

import vedo


if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson"

    filter_dataset = "replica_v1"

    dataset_path = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test_place_picker", "sampled_it_clearance")
    samples_it_optim_down_dir = opj(base_dir, "test_place_picker", "sampled_it_clearance_opti_down_trans")
    samples_place_dir = opj(base_dir, "test_place_picker", "sampled_place_exec")

    interactions_by_type = {
        "laying": ["laying_bed", "laying_hands_up", "laying_on_sofa", "laying_sofa_foot_on_floor"],
        "reaching_out": ["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table",
                         "reachin_out_ontable_one_hand"],
        "sitting": ["sitting", "sitting_bit_open_arms", "sitting_chair", "sitting_comfortable", "sitting_compact",
                    "sitting_hands_on_device", "sitting_looking_to_right", "sitting_small_table", "sitting_stool",
                    "sitting_stool_one_foot_floor"],
        "standing_up": ["standing_up", "standup_hand_on_furniture"],
        "walking": ["walking_left_foot", "walking_right_foot"]
    }


    follow_up_file = opj(base_dir,'test_place_picker', 'follow_up_process.csv')
    last_task_colum = "num_it_picked_sampled_opti_down_trans"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1])

    completed_task =[]
    for dataset_name, scene_name in follow_up_data.index:
        final = follow_up_data.at[(dataset_name, scene_name), last_task_colum]
        for num_point in range(0, final):
            completed_task.append((dataset_name, scene_name, num_point))

    num_total_task = follow_up_data['goal_place_picked_sampled'].sum()

    print('COMPLETED TASKS: total %d, done %d' % (num_total_task, len(completed_task)))

    random.shuffle(completed_task)

    for dataset_name, scene_name, num_point in completed_task:

        if filter_dataset is not None and filter_dataset != dataset_name:
            continue

        print(dataset_name, scene_name, num_point)

        # extracting information about type of interaction on PLACE execution
        interaction_type_df = pd.read_csv(opj(samples_place_dir, scene_name, "interactions.txt"), index_col=0, header=None)
        interaction_type = interaction_type_df.at[num_point, 1]

        available_interactions=[]
        for current_interaction in interactions_by_type[interaction_type]:
            file_name = f"body_{num_point}_{current_interaction}.ply"
            if os.path.exists(opj(samples_it_dir, scene_name, file_name)):
                available_interactions.append((current_interaction, file_name))


        vedo_scene = vedo.load(opj(dataset_path, dataset_name, "scenes", scene_name + ".ply"))
        vedo_scene.backFaceCulling(value=True)

        plt = vedo.Plotter(N=len(available_interactions)+3,
                           title=f"{dataset_name}/{scene_name}", size=(1800, 1000), axes=4)

        place_orig = vedo.load(opj(samples_place_dir, scene_name, f"body_{num_point}_orig.ply")).color("white")
        place_opt1 = vedo.load(opj(samples_place_dir, scene_name, f"body_{num_point}_opt1.ply")).color("white")
        place_opt2 = vedo.load(opj(samples_place_dir, scene_name, f"body_{num_point}_opt2.ply")).color("white")
        plt.show(vedo_scene + place_orig, "PLACE, No optimization", at=0)
        plt.show(vedo_scene + place_opt1, "PLACE SimOptim", at=1)
        plt.show(vedo_scene + place_opt2, "PLACE AdvOptim", at=2)

        i=0
        for current_interaction, file_name in available_interactions:
            # it_b0 = vedo.load(opj(samples_it_dir, scene_name, file_name)).color("yellow").alpha(.5)
            it_b0_opti_down = vedo.load(opj(samples_it_optim_down_dir, scene_name, file_name)).color("green")
            plt.show(vedo_scene+it_b0_opti_down, f"iTClearance {current_interaction}", at=i+3)
            # plt.show(vedo_scene+,f"{current_interaction} OptiDown", at=i*2+4)
            i+=1
        vedo.interactive()

        plt.close()