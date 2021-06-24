import random

import pandas as pd
from  os.path import join as opj

import vedo


if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    filter_dataset = "mp3d"

    dataset_path = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test", "sampled_it_clearance")
    samples_place_dir = opj(base_dir, "test", "sampled_place_exec")

    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    place_follow_up_column = "place_samples_extracted"
    itC_follow_up_column = "it_samples_extracted"
    current_follow_up_column ='turk_sample_extracted'

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[ (follow_up_data[current_follow_up_column] == False)
                                         &  (follow_up_data[place_follow_up_column]==True)
                                         &  (follow_up_data[itC_follow_up_column]==True)].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[ (follow_up_data[current_follow_up_column] == True)
                                         &  (follow_up_data[place_follow_up_column]==True)
                                         &  (follow_up_data[itC_follow_up_column]==True)].index.size

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    random.shuffle(pending_tasks)
    for dataset_name, scene_name, interaction in pending_tasks:

        if filter_dataset is not None and filter_dataset != dataset_name:
            continue

        vedo_scene = vedo.load(opj(dataset_path, dataset_name, "scenes", scene_name+".ply"))
        vedo_scene.backFaceCulling(value=True)
        place_b0_orig = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_0_orig.ply" )).color("white")
        place_b0_opt1 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_0_opt1.ply" )).color("white")
        place_b0_opt2 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_0_opt2.ply" )).color("white")
        place_b1_orig = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_1_orig.ply" )).color("white")
        place_b1_opt1 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_1_opt1.ply" )).color("white")
        place_b1_opt2 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_1_opt2.ply" )).color("white")
        place_b2_orig = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_2_orig.ply" )).color("white")
        place_b2_opt1 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_2_opt1.ply" )).color("white")
        place_b2_opt2 = vedo.load(opj(samples_place_dir, scene_name, interaction, "body_2_opt2.ply" )).color("white")
        it_b0 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_0.ply" ))
        it_b1 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_1.ply" ))
        it_b2 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_2.ply" ))

        plt = vedo.Plotter(shape=(3,4), title=f"{dataset_name}/{scene_name}", size=(1024,1080), axes=0)

        plt.show(vedo_scene+place_b0_orig, at=0)
        plt.show(vedo_scene+place_b0_opt1, at=1)
        plt.show(vedo_scene+place_b0_opt2, at=2)
        plt.show(vedo_scene+it_b0, at=3)

        plt.show(vedo_scene+place_b1_orig, at=4)
        plt.show(vedo_scene+place_b1_opt1, at=5)
        plt.show(vedo_scene+place_b1_opt2, at=6)
        plt.show(vedo_scene+it_b1, at=7)

        plt.show(vedo_scene+place_b2_orig, at=8)
        plt.show(vedo_scene+place_b2_opt1, at=9)
        plt.show(vedo_scene+place_b2_opt2, at=10)
        plt.show(vedo_scene+it_b2, at=11 )
        vedo.interactive()

        plt.close()