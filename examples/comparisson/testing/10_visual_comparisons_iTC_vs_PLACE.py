import random

import pandas as pd
from  os.path import join as opj

import trimesh
import vedo


if __name__ == '__main__':

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson"
    view_place_demo_conf = True
    full_visualization = False
    shuffle_order = True

    filter_dataset = None # None, mp3d, prox, replica_v1
    filter_scene = None # "frl_apartment_0"
    filter_interaction = None #"reaching_out_ontable_one_hand"

    dataset_path = opj(base_dir, "datasets")

    samples_it_dir = opj(base_dir, "test", "sampled_it_clearance")
    samples_it_optim_smplx_dir = opj(base_dir, "test", "sampled_it_clearance_opti_smplx")

    if view_place_demo_conf:
        samples_place_dir = opj(base_dir, "test", "sampled_place_exec[demo_conf]")
    else:
        samples_place_dir = opj(base_dir, "test", "sampled_place_exec")


    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    if view_place_demo_conf:
        place_follow_up_column = "place_auto_samples_extracted[demo_conf]"
    else:
        place_follow_up_column = "place_auto_samples_extracted"
    itC_follow_up_column = "it_auto_samples"


    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])

    num_total_task = follow_up_data.index.size
    completed_task = list(follow_up_data[(follow_up_data[place_follow_up_column] == True)
                                         & (follow_up_data[itC_follow_up_column]==True)].index)

    num_completed_task = len(completed_task)

    print( 'COMPLETED TASKS: total %d, done %d' % (num_total_task, num_completed_task))

    if shuffle_order:
        random.shuffle(completed_task)
    else:
        completed_task.sort()
    for dataset_name, scene_name, interaction in completed_task:
        print(dataset_name, "   ", scene_name, "   ", interaction)
        if filter_dataset is not None and filter_dataset != dataset_name:
            continue
        if filter_scene is not None and filter_scene != scene_name:
            continue
        if filter_interaction is not None and filter_interaction != interaction:
            continue

        vedo_scene = vedo.load(opj(dataset_path, dataset_name, "scenes", scene_name+".ply"))
        vedo_scene.backFaceCulling(value=True)

        samples_place_subdir = opj(samples_place_dir, scene_name, interaction)
        num_it_auto_samples = follow_up_data.loc[(dataset_name, scene_name, interaction)]["num_it_auto_samples"]

        place_b0_opt2 = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_0_opt2.ply"))).color("white")
        it_b0_opti_smplx = vedo.load(opj(samples_it_optim_smplx_dir, scene_name, interaction, "body_0.ply" )).color("green")
        if num_it_auto_samples>1:
            place_b1_opt2 = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_1_opt2.ply" ))).color("white")
            it_b1_opti_smplx = vedo.load(opj(samples_it_optim_smplx_dir, scene_name, interaction, "body_1.ply")).color("green")
        if num_it_auto_samples > 2:
            place_b2_opt2 = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_2_opt2.ply" ))).color("white")
            it_b2_opti_smplx = vedo.load(opj(samples_it_optim_smplx_dir, scene_name, interaction, "body_2.ply" )).color("green")

        if full_visualization:
            place_b0_orig = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_0_orig.ply" ))).color("white")
            place_b0_opt1 =vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_0_opt1.ply" ))).color("white")
            it_b0 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_0.ply" ))
            if num_it_auto_samples > 1:
                place_b1_orig = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_1_orig.ply" ))).color("white")
                place_b1_opt1 = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_1_opt1.ply" ))).color("white")
                it_b1 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_1.ply" ))
            if num_it_auto_samples > 2:
                place_b2_orig = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_2_orig.ply" ))).color("white")
                place_b2_opt1 = vedo.trimesh2vtk(trimesh.load(opj(samples_place_subdir, "body_2_opt1.ply" ))).color("white")
                it_b2 = vedo.load(opj(samples_it_dir, scene_name, interaction, "body_2.ply" ))



        if full_visualization:
            plt = vedo.Plotter(shape=(3,5), title=f"{dataset_name}/{scene_name}/{interaction}", size=(1800,1000), axes=4)

            plt.show(vedo_scene+place_b0_orig, "PLACE, No optimization" ,at=0)
            plt.show(vedo_scene+place_b0_opt1, "PLACE SimOptim", at=1)
            plt.show(vedo_scene+place_b0_opt2, "PLACE AdvOptim", at=2)
            plt.show(vedo_scene+it_b0,"iTCleareance", at=3)
            plt.show(vedo_scene+it_b0_opti_smplx,"iTCleareance optimized", at=4)

            if num_it_auto_samples > 1:
                plt.show(vedo_scene+place_b1_orig, at=5)
                plt.show(vedo_scene+place_b1_opt1, at=6)
                plt.show(vedo_scene+place_b1_opt2, at=7)
                plt.show(vedo_scene+it_b1, at=8)
                plt.show(vedo_scene+it_b1_opti_smplx, at=9)
            if num_it_auto_samples > 2:
                plt.show(vedo_scene+place_b2_orig, at=10)
                plt.show(vedo_scene+place_b2_opt1, at=11)
                plt.show(vedo_scene+place_b2_opt2, at=12)
                plt.show(vedo_scene+it_b2, at=13 )
                plt.show(vedo_scene+it_b2_opti_smplx, at=14 )
        else:
            plt = vedo.Plotter(shape=(2, 3), title=f"{dataset_name}/{scene_name}/{interaction}", size=(1800, 1000), axes=4)

            plt.show(vedo_scene + place_b0_opt2, "PLACE AdvOptim", at=0)
            plt.show(vedo_scene + it_b0_opti_smplx, "iTCleareance optimized", at=3)
            if num_it_auto_samples > 1:
                plt.show(vedo_scene + place_b1_opt2, "PLACE AdvOptim", at=1)
                plt.show(vedo_scene + it_b1_opti_smplx, "iTCleareance optimized", at=4)
            if num_it_auto_samples > 2:
                plt.show(vedo_scene + place_b2_opt2, "PLACE AdvOptim", at=2)
                plt.show(vedo_scene + it_b2_opti_smplx, "iTCleareance optimized", at=5)


        vedo.interactive()

        plt.close()