import math
import numpy as np
import smplx
import trimesh
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from vedo import trimesh2vtk
import chamfer_pytorch.dist_chamfer as ext

from human_body_prior.tools.model_loader import load_vposer
from preprocess.bps_encoding import bps_gen_ball_inside, bps_encode_scene
from util.util_preprocessing import crop_scene_sphere_smplx_at_point
from utils import update_globalRT_for_smplx, gen_body_mesh, convert_to_3D_rot, get_contact_id, convert_to_6D_rot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vposer_model(vposer_model_path, vp_model='snapshot'):
    vposer_model, _ = load_vposer(vposer_model_path, vp_model=vp_model)
    vposer_model.to(device)
    return vposer_model

def load_smplx_model(smplx_model_path, gender):
    smplx_model = smplx.create(smplx_model_path, model_type='smplx',
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
                               create_transl=True,
                               batch_size=1
                               ).to(device)

    return smplx_model


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


def optimize_body_on_environment(scene_trimesh, s_grid_min_batch, s_grid_max_batch, s_sdf_batch,
                                 smplx_model, vposer_model,
                                 np_body_params, np_point, best_angle, contact_ids,
                                 weight_loss_rec_verts=1.0,
                                 weight_loss_rec_bps=1.0,
                                 weight_loss_vposer=0.02,
                                 weight_loss_shape=0.01,
                                 weight_loss_hand=0.01,
                                 weight_collision=8.0,
                                 weight_loss_contact=0.5,
                                 itr_s2=150,
                                 view_evolution_screens=True):
    np_body_params = rotate_smplx_body(np_body_params, smplx_model, best_angle)
    np_body_params = translate_smplx_body(np_body_params, smplx_model, np_point)

    body_trimesh_proxd = get_trimesh_from_body_params(smplx_model, vposer_model, np_body_params)

    selected_p = body_trimesh_proxd.bounding_sphere.centroid

    print('[INFO] Position selected.')

    scene_verts = scene_trimesh.vertices

    cube_size = body_trimesh_proxd.bounding_sphere.extents[0] * 1.2
    r = cube_size / 2
    scene_verts_local, scene_verts_crop_local, shift = crop_scene_sphere_smplx_at_point(scene_verts, selected_p, r=r)

    print('[INFO] scene mesh cropped and shifted.')

    scene_basis_set = bps_gen_ball_inside(n_bps=10000, random_seed=100)

    scene_verts_crop_global = scene_verts_crop_local / cube_size

    scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set,
                                                                            scene_verts_crop_global)  # [n_feat, n_bps]
    selected_scene_verts_local = scene_verts_crop_local[selected_ind]

    print('[INFO] bps encoding computed.')

    # position body params in the new reference point "shift"
    shifted_rotated_scene = trimesh.Trimesh(vertices=scene_verts_local, faces=scene_trimesh.faces)
    shifted_rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors

    body_params = translate_smplx_body(np_body_params, smplx_model, shift)

    if view_evolution_screens:
        shifted_rotated_body_smplx_trimesh = get_trimesh_from_body_params(smplx_model, vposer_model,
                                                                          body_params)
        s = trimesh.Scene()
        s.add_geometry(shifted_rotated_scene)
        s.add_geometry(shifted_rotated_body_smplx_trimesh)
        s.add_geometry(trimesh.points.PointCloud(selected_scene_verts_local, colors=[0, 255, 0]))
        s.show(caption="smplx_model shifted and rotated", flags={'axis': True})

    # extract body params of body positioned at "shift" and scaled to cube size
    np_body_verts_sample = get_vertices_from_body_params(smplx_model, vposer_model, body_params) / cube_size
    body_verts_sample = torch.from_numpy(np_body_verts_sample).float().unsqueeze(0).to(device).permute(0, 2, 1)

    nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(np_body_verts_sample)
    np_body_bps_sample, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
    body_bps_sample = torch.from_numpy(np_body_bps_sample).float().unsqueeze(0).unsqueeze(0).to(device)

    if view_evolution_screens:
        body_trimesh_sampled = trimesh.Trimesh(np_body_verts_sample * cube_size, faces=smplx_model.faces)
        body_trimesh_sampled.visual.face_colors = [0, 255, 255, 100]
        s = trimesh.Scene()
        s.add_geometry(shifted_rotated_scene)
        s.add_geometry(body_trimesh_sampled)
        s.show(caption="body sampled (verifying scale)", flags={'axis': True})

    # initialize smplx params, bs=1, local 3D cage coordinate system
    body_params_rec = torch.from_numpy(body_params).float().unsqueeze(0).to(device)
    body_params_rec = convert_to_6D_rot(body_params_rec)
    body_params_rec.requires_grad = True

    print('[INFO] start optimization ...')
    optimizer = optim.Adam([body_params_rec], lr=0.01)

    body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
    body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

    for step in tqdm(range(itr_s2)):
        optimizer.zero_grad()

        body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
        body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]

        # transform body verts to unit ball global coordinate
        body_verts_rec_global = body_verts_rec / cube_size  # scale into unit ball

        # calculate body_bps_rec
        body_bps_rec = torch.zeros(body_bps_sample.shape)
        if weight_loss_rec_bps > 0:
            nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
                body_verts_rec_global.detach().cpu().numpy())
            neigh_dist, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
            body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - torch.from_numpy(
                selected_scene_verts_global).float().to(device)  # [n_bps, 3]
            body_bps_rec = torch.sqrt(
                body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(
                0)  # [bs=1, 1, n_bps]

        ### body bps encoding reconstruct loss
        loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
        loss_rec_bps = F.l1_loss(body_bps_sample, body_bps_rec)

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

        # transfrom body_verts_rec (local 3d cage coordinate system) to prox coordinate system
        temp = body_verts_rec - torch.from_numpy(shift).float().to(device)
        body_verts_rec_prox = temp.unsqueeze(0)  # tensor, [bs=1, 10475, 3]

        ### sdf collision loss
        norm_verts_batch = (body_verts_rec_prox - s_grid_min_batch) / (
                s_grid_max_batch - s_grid_min_batch) * 2 - 1
        n_verts = norm_verts_batch.shape[1]
        body_sdf_batch = F.grid_sample(s_sdf_batch.unsqueeze(1),
                                       norm_verts_batch[:, :, [2, 1, 0]].view(-1, n_verts, 1, 1, 3),
                                       padding_mode='border')
        # if there are no penetrating vertices then set sdf_penetration_loss = 0
        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_collision = torch.tensor(0.0, dtype=torch.float32).to(device)
        else:
            loss_collision = body_sdf_batch[body_sdf_batch < 0].abs().mean()

        ### contact loss
        body_verts_contact = body_verts_rec.unsqueeze(0)[:, contact_ids, :]  # [1,1121,3]
        dist_chamfer_contact = ext.chamferDist()
        # scene_verts: [bs=1, n_scene_verts, 3]
        scene_verts = torch.from_numpy(scene_verts_local).float().to(device).unsqueeze(0)  # [1,50000,3]
        contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(), scene_verts.contiguous())
        loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))

        loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand + weight_collision * loss_collision + weight_loss_contact * loss_contact
        loss.backward(retain_graph=True)
        optimizer.step()

    print('[INFO] optimization stage finished.')

    body_params_opt_s2 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
    np_body_params_optim = translate_smplx_body(body_params_opt_s2.detach().squeeze().cpu().numpy(), smplx_model,
                                                -shift)
    body_trimesh_optim = get_trimesh_from_body_params(smplx_model, vposer_model, np_body_params_optim)
    body_trimesh_optim.visual.face_colors = [255, 255, 255, 255]

    return body_trimesh_optim, np_body_params_optim



if __name__ == '__main__':
    r_name="MPH112_00150_01"
    print(get_scene_name_from_proxd_recording_id(r_name))
