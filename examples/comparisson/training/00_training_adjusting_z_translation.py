"""
 This helps to determine if a translation for adjusting the body mesh was performed on the old training
"""
import json
import os
import shutil
from os.path import join as opj

import numpy as np
import trimesh
import pandas as pd

from util.util_proxd import load_smplx_model, translate_smplx_body, get_vertices_from_body_params, load_vposer_model

if __name__ == "__main__":

    basis_dir = "output"

    first_trainings_dir =  f"{basis_dir}/descriptors_repository_v1"
    output_dir = f"{basis_dir}/descriptors_repository_v2"

    datasets_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data"
    smplx_model_path = opj(datasets_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_path = opj(datasets_dir, "pretrained_place", "body_models", "vposer_v1_0")

    df = pd.read_csv(f"{basis_dir}/v1_to_v2_z_translation.csv", index_col=0)

    for interaction_name in os.listdir(first_trainings_dir):

        z_translation = float(df[df["interaction"]==interaction_name]["z_translation"])

        input_subdir = opj(first_trainings_dir, interaction_name)
        output_subdir = opj(output_dir, interaction_name)

        if z_translation == 0:
            shutil.copytree(input_subdir, output_subdir)
        else:
            obj_file_name = [f for f in os.listdir(opj(first_trainings_dir,  interaction_name)) if f.endswith("_object.ply")][0]

            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            print(interaction_name.upper())
            print(f"      z_translation: {z_translation}")

            prefix_file_name = obj_file_name[:obj_file_name.find("_object.ply")]

            src_obj_file = opj(input_subdir, prefix_file_name + "_object.ply")
            dst_obj_file = opj(output_subdir, prefix_file_name + "_object.ply")
            new_body_mesh = trimesh.load(src_obj_file)
            new_body_mesh.apply_translation([0, 0, z_translation])
            new_body_mesh.export(dst_obj_file)

            src_env_file = opj(input_subdir,prefix_file_name+"_environment.ply")
            dst_env_file = opj(output_subdir, prefix_file_name + "_environment.ply")
            shutil.copy(src_env_file, dst_env_file)

            src_json_file = opj(input_subdir, prefix_file_name + ".json")
            dst_json_file = opj(output_subdir, prefix_file_name + "json")
            with open(src_json_file, 'r') as f:
                json_training_data = json.load(f)
            json_training_data["extra"]["manual_z_translation"] = z_translation
            with open(dst_json_file, 'w') as fp:
                json.dump(json_training_data, fp, indent=4, sort_keys=True)


            np_body_params_name = prefix_file_name + "_smplx_body_params.npy"
            np_body_params = np.load(opj(input_subdir, np_body_params_name))
            smplx_model = load_smplx_model(smplx_model_path, json_training_data["extra"]["body_gender"])
            np_body_params = translate_smplx_body(np_body_params, smplx_model, [0, 0, z_translation])
            np.save(opj(output_subdir, np_body_params_name), np_body_params)

            # #### VISUALIZATION
            s = trimesh.Scene()
            env_mesh = trimesh.load(dst_env_file)
            s.add_geometry(env_mesh)
            s.add_geometry(new_body_mesh)
            vposer_model = load_vposer_model(vposer_model_path)
            np_body_verts_sample = get_vertices_from_body_params(smplx_model, vposer_model, np_body_params)
            body_trimesh_proxd = trimesh.Trimesh(np_body_verts_sample, faces=smplx_model.faces)
            body_trimesh_proxd.visual.face_colors = [255, 255, 255, 255]
            s.add_geometry(body_trimesh_proxd)

            s.show()