import smplx
import torch
import math
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from human_body_prior.tools.model_loader import load_vposer
from preprocess.bps_encoding import bps_gen_ball_inside, bps_encode_scene, bps_encode_body
from preprocess.preprocess_optimize import augmentation_crop_scene_smplx
from util.util_preprocessing import crop_scene_cube_smplx_at_point
from utils import convert_to_6D_rot, convert_to_3D_rot, gen_body_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_smplx_model(model_folder, gender):
    smplx_model = smplx.create(model_folder, model_type='smplx',
                               gender=gender, ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True
                               )
    smplx_model.to(device)
    return smplx_model


def get_scaledShifted_bps_sets(np_body_verts_sample: np.ndarray, np_scene_verts:np.array, cube_size = 2):
    p_cage_center = np.average(np_body_verts_sample, axis=0)

    _, scene_verts_crop, shift = crop_scene_cube_smplx_at_point(
        scene_verts=np_scene_verts, scene_center=p_cage_center, r=cube_size, with_wall_ceilling=True)

    body_verts_global = (np_body_verts_sample + shift) / cube_size
    scene_verts_crop_scaled = scene_verts_crop / cube_size

    bps = bps_gen_ball_inside(n_bps=10000, random_seed=100)
    scene_bps_feat, scene_bps_verts, selected_ind = bps_encode_scene(bps, scene_verts_crop_scaled)

    body_bps_feat, body_bps_verts, selected_body_ind = bps_encode_scene(scene_bps_verts, body_verts_global)

    return scene_verts_crop_scaled, bps, scene_bps_feat, scene_bps_verts,body_bps_feat, body_bps_verts, shift

# optimizer_stage_1
def adjust_body_mesh_to_raw_guess(np_body_verts_sample: np.ndarray, np_scene_verts:np.array,
                                  vposer_model, smplx_model,
                                  weight_loss_rec_verts=1.0, weight_loss_rec_bps = 3.0,
                                  weight_loss_vposer = 0.02, weight_loss_shape = 0.01,
                                  weight_loss_hand = 0.01,
                                  itr_s1 = 300, cube_size = 2):

    scene_verts_crop_scaled, bps, scene_bps_feat, scene_bps_verts,body_bps_feat, body_bps_verts, shift = get_scaledShifted_bps_sets(np_body_verts_sample, np_scene_verts, cube_size)

    body_params_rec = torch.randn(1, 72).to(device)  # initialize smplx params, bs=1, local 3D cage coordinate system
    body_params_rec[0, 0] = 0.0
    body_params_rec[0, 1] = 0.0
    body_params_rec[0, 2] = 0.0
    body_params_rec[0, 3] = 1.5
    body_params_rec[0, 4] = 0.0
    body_params_rec[0, 5] = 0.0
    body_params_rec = convert_to_6D_rot(body_params_rec)
    body_params_rec.requires_grad = True
    optimizer = optim.Adam([body_params_rec], lr=0.1)

    np_body_verts_scaled = (np_body_verts_sample + shift) / cube_size
    body_verts = torch.tensor(np_body_verts_scaled[np.newaxis, :, :]).float().to(device) # [1, 10475, 3]
    body_bps = torch.from_numpy(body_bps_feat[np.newaxis,:, :]).float().to(device)

    for step in tqdm(range(itr_s1)):
        if step > 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
        if step > 300:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        optimizer.zero_grad()

        body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
        body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]

        # transform body verts to unit ball global coordinate system
        body_verts_rec_scaled = body_verts_rec / cube_size  # scale into unit ball

        # calculate optimized body bps feature
        body_bps_rec = torch.zeros(body_bps_feat.shape)
        if weight_loss_rec_bps > 0:
            nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
                body_verts_rec_scaled.detach().cpu().numpy())
            neigh_dist, neigh_ind = nbrs.kneighbors(scene_bps_verts)
            body_bps_rec = body_verts_rec_scaled[neigh_ind.squeeze()] - torch.from_numpy(
                scene_bps_verts).float().to(device)  # [n_bps, 3]
            body_bps_rec = torch.sqrt(
                body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(
                0)  # [bs=1, 1, n_bps]

        ### body bps feature reconstruct loss
        loss_rec_verts = torch.nn.functional.l1_loss(body_verts, body_verts_rec.unsqueeze(0))
        loss_rec_bps = torch.nn.functional.l1_loss(body_bps, body_bps_rec)

        ### vposer loss
        body_params_rec_72 = convert_to_3D_rot(body_params_rec)
        vposer_pose = body_params_rec_72[:, 16:48]
        loss_vposer = torch.mean(vposer_pose ** 2)
        ### shape prior loss
        shape_params = body_params_rec_72[:, 6:16]
        loss_shape = torch.mean(shape_params ** 2)
        ### hand pose prior loss
        hand_params = body_params_rec_72[:, 48:]
        loss_hand = torch.mean(hand_params ** 2)

        loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand
        loss.backward(retain_graph=True)
        optimizer.step()
    return body_params_rec, shift