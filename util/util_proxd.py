import math
import numpy as np
import trimesh
import torch

from utils import update_globalRT_for_smplx, gen_body_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate_smplx_body(np_body_params,smplx_model, shift):
    trans_matrix_2 = np.array([[1, 0, 0, shift[0]],
                                      [0, 1, 0, shift[1]],
                                       [0, 0, 1, shift[2]],
                                       [0, 0, 0, 1]])
    body_params = update_globalRT_for_smplx(np_body_params, smplx_model, trans_matrix_2)
    return body_params


def rotate_smplx_body(np_body_params,smplx_model, z_rotation):
    trans_matrix_1 = np.array([[math.cos(z_rotation), -math.sin(z_rotation), 0, 0],
                                       [math.sin(z_rotation), math.cos(z_rotation), 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
    body_params = update_globalRT_for_smplx(np_body_params, smplx_model, trans_matrix_1)
    return body_params

def get_trimesh_from_body_params(smplx_model, vposer_model, body_params):
    body_verts_p = get_vertices_from_body_params(smplx_model, vposer_model, body_params)
    body_trimesh_p = trimesh.Trimesh(vertices=body_verts_p, faces=smplx_model.faces, face_colors=[200, 200, 0, 255])

    return body_trimesh_p

def get_vertices_from_body_params(smplx_model, vposer_model, body_params):
    body_params = torch.from_numpy(body_params).unsqueeze(0).to(device).float()
    body_pose_joint_p = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(1, -1)
    body_verts_p = gen_body_mesh(body_params, body_pose_joint_p, smplx_model)[0].detach().cpu().numpy()

    return body_verts_p




def get_scene_name_from_proxd_recording_id(proxd_recording_id:str):
    return proxd_recording_id[:proxd_recording_id.find("_")]


if __name__ == '__main__':
    r_name="MPH112_00150_01"
    print(get_scene_name_from_proxd_recording_id(r_name))
