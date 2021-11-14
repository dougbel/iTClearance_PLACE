"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import warnings
warnings.simplefilter("ignore", UserWarning)
from os.path import  join as opj

import torch
from vedo import load

from it_clearance.testing.tester import TesterClearance
from util.util_mesh import read_sdf
import pandas as pd
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")

    n_sample=4800

    filter_dataset = "prox"    # None   prox   mp3d  replica_v1

    json_conf_execution_dir = opj(base_dir,"config", "json_execution")
    directory_of_prop_configs= opj(base_dir, "config","propagators_configs")
    directory_of_trainings = opj(base_dir, "config", "descriptors_repository")
    datasets_dir = opj(base_dir, "datasets")

    env_filled_data_test_dir = opj(output_base, "bubble_fillers")
    env_raw_data_test_dir = opj(output_base, "no_bubble_fillers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    filles_to_test={
        "conglo_env_raw_iT_naive": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_raw_iT_clearance": opj(env_raw_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv"),
        # "conglo_env_fill_iT_naive": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_naive_{filter_dataset}.csv"),
        "conglo_env_fill_iT_clearance": opj(env_filled_data_test_dir, f"02_conglomerate_capable_positions_clearance_{filter_dataset}.csv")
    }

    loss_non_collision_sample, loss_contact_sample = 0, 0

    column_prefix = f"ablation_{filter_dataset}_"
    for file in filles_to_test:
        conglo_path =filles_to_test[file]
        print(conglo_path)

        conglo_data = pd.read_csv(conglo_path)

        n_sampling = sum([x.startswith(column_prefix) for x in conglo_data.columns.to_list()])+1
        follow_up_column = f"{column_prefix}{n_sampling}"
        sample = conglo_data.sample(n_sample)

        for idx, row in sample.iterrows():
            dataset = row["dataset"]
            env_name = row["scene"]
            interaction = row["interaction"]
            angle = row["angle"]
            json_conf_execution_file = opj(json_conf_execution_dir, f"single_testing_{interaction}.json")
            tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
            subdir_name = "_".join(tester.affordances[0])
            ply_obj_file = opj(directory_of_trainings, interaction, subdir_name + "_object.ply")
            vtk_object = load(ply_obj_file)
            vtk_object.rotate(angle, axis=(0, 0, 1), rad=True)
            vtk_object.pos(x=row["point_x"], y=row["point_y"], z=row["point_z"])

            s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_sdf(opj(datasets_dir, dataset), env_name)

            #####################  compute non-collision/contact score ##############
            # body verts before optimization
            # [1, 10475, 3]
            body_verts_sample = np.asarray(vtk_object.points())
            body_verts_sample_prox_tensor = torch.from_numpy(body_verts_sample).float().unsqueeze(0).to(device)
            norm_verts_batch = (body_verts_sample_prox_tensor - s_grid_min_batch) / (
                        s_grid_max_batch - s_grid_min_batch) * 2 - 1
            body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                           norm_verts_batch[:, :, [2, 1, 0]].view(-1, 10475, 1, 1, 3),
                                           padding_mode='border')
            if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
                loss_non_collision_sample += 1.0
                loss_contact_sample += 0.0
            else:
                loss_non_collision_sample += (body_sdf_batch > 0).sum().float().item() / 10475.0
                loss_contact_sample += 1.0

        loss_non_collision_sample = loss_non_collision_sample / n_sample
        loss_contact_sample = loss_contact_sample / n_sample
        print('w/o optimization body: non_collision score:', loss_non_collision_sample)
        print('w/o optimization body: contact score:', loss_contact_sample)



        conglo_data[follow_up_column] = False
        conglo_data.loc[sample.index.to_list(),[follow_up_column]] =True
        conglo_data.to_csv(conglo_path,index=False)