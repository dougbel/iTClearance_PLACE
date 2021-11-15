"""
It count an generate the number of point detected that facilitates a given afordances.
Creates a csv file with a resume of the data
"""
import statistics
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

from tabulate import tabulate

def get_next_sampling_id(l_column_names):
    return max([int(x.replace(column_prefix, "")) for x in l_column_names if
         x.startswith(column_prefix) and x.replace(column_prefix, "").isdigit()]) + 1

if __name__ == '__main__':
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    output_base = opj(base_dir, "ablation_study_in_test")

    stratified_sampling = True

    # n_sample_per_scene=1297 # confidence level = 97%, margin error = 3%  for infinite samples
    n_sample_per_scene=10 #

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


    column_prefix = f"ablation_{filter_dataset}_"
    for model in filles_to_test:
        tb_headers = ["model", "dataset", "scene", "non_collision","std_dev", "contact"]
        tb_data = []
        conglo_path =filles_to_test[model]
        print(conglo_path)

        loss_non_collisions_model=[]
        loss_contacts_model=[]

        conglo_data = pd.read_csv(conglo_path)
        n_sampling = get_next_sampling_id(conglo_data.columns.to_list())
        follow_up_column = f"{column_prefix}{n_sampling}"
        conglo_data[follow_up_column] = False
        conglo_data[follow_up_column + "non_collision"] = ""
        conglo_data[follow_up_column + "contact_sample"] = ""

        grouped = conglo_data.groupby(conglo_data['scene'])

        for current_env_name in conglo_data['scene'].unique():

            loss_non_collision_env, loss_contact_env = [], []

            dataset_results = grouped.get_group(current_env_name)
            if stratified_sampling:
                sample = dataset_results.groupby('interaction_type', group_keys=False).apply(lambda x: x.sample(int(np.rint(n_sample_per_scene*len(x)/len(dataset_results))))).sample(frac=1)
            else:
                sample = conglo_data.sample(n_sample_per_scene)
            sample[follow_up_column + "non_collision"] = 0.0
            sample[follow_up_column + "contact_sample"] = 0.0
            # sample = conglo_data.sample(n_sample)

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

                current_loss_non_coll = 0.0
                current_loss_contact = 0.0
                if body_sdf_batch.lt(0).sum().item() < 1:  # if no interpenetration: negative sdf entries is less than one
                    current_loss_non_coll= 1.0
                    current_loss_contact = 0.0
                else:
                    current_loss_non_coll = (body_sdf_batch > 0).sum().float().item() / 10475.0
                    current_loss_contact = 1.0

                sample.loc[idx, [follow_up_column + "non_collision"]] = current_loss_non_coll
                sample.loc[idx, [follow_up_column + "contact_sample"]] = current_loss_contact
                loss_non_collision_env.append( current_loss_non_coll)
                loss_contact_env.append(current_loss_contact)

                loss_non_collisions_model.append(current_loss_non_coll)
                loss_contacts_model.append(current_loss_contact)

            # loss_non_collision_env = loss_non_collision_env / n_sample_per_scene
            # loss_contact_env = loss_contact_env / n_sample_per_scene
            # print("   Scene", current_env_name)
            # print('      non_collision score:', loss_non_collision_env)
            # print('      contact score:', loss_contact_env)
            tb_data.append([model, filter_dataset, current_env_name, statistics.mean(loss_non_collision_env), statistics.stdev(loss_non_collision_env),  statistics.mean(loss_contact_env) ])

            conglo_data.loc[sample.index.to_list(),[follow_up_column]] =True
            conglo_data.loc[sample.index.to_list(),[follow_up_column + "non_collision"]] = sample.loc[sample.index.to_list(),[follow_up_column + "non_collision"]]
            conglo_data.loc[sample.index.to_list(),[follow_up_column + "contact_sample"]] = sample.loc[sample.index.to_list(),[follow_up_column + "contact_sample"]]
        # print("  Overall")
        #collision_score =  # sum(loss_non_collisions_model)/len(loss_non_collisions_model)
        #contact_score =  # sum(loss_contacts_model)/len(loss_contacts_model)
        # print('      non_collision score:', collision_score)
        # print('      contact score:', contact_score)
        tb_data.append([model, filter_dataset, "Overall", statistics.mean(loss_non_collisions_model), statistics.stdev(loss_non_collisions_model),  statistics.mean(loss_contacts_model)])

        print(tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        print(tabulate(tb_data, headers=tb_headers, floatfmt=".4f", tablefmt="simple"))

        import logging

        logging.basicConfig(filename=f"output_{follow_up_column}.txt", level=logging.DEBUG, format='')

        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="latex_booktabs"))
        logging.info('\n'+tabulate(tb_data,headers=tb_headers, floatfmt=".4f",  tablefmt="simple"))


        # conglo_data.to_csv(conglo_path,index=False)