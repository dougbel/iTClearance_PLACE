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


# optimizer_stage_1
def adjust_body_mesh_to_raw_guess(np_body_verts_sample: np.ndarray, np_scene_verts:np.array,
                                  vposer_model, smplx_model,
                                  weight_loss_rec_verts=1.0, weight_loss_rec_bps = 3.0,
                                  weight_loss_vposer = 0.02, weight_loss_shape = 0.01,
                                  weight_loss_hand = 0.01,
                                  itr_s1 = 300, cube_size = 2, vedo_env=None, body_faces=None):

    p_cage_center = np.average(np_body_verts_sample, axis=0)

    scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_at_point(
        scene_verts = np_scene_verts, scene_center=p_cage_center, r=cube_size, with_wall_ceilling=True)

    body_verts_global = (np_body_verts_sample + shift) / cube_size
    scene_verts_crop_global = scene_verts_crop_local / cube_size


    bps = bps_gen_ball_inside(n_bps=10000, random_seed=100)
    scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(bps, scene_verts_crop_global)

    body_bps, selected_body_verts_global, selected_body_ind = bps_encode_scene(selected_scene_verts_global,
                                                                                      body_verts_global)  # [n_feat, n_bps]

    import trimesh
    import vedo
    trimesh_env=vedo.vtk2trimesh(vedo_env)
    trimesh_body=trimesh.Trimesh(np_body_verts_sample,body_faces)
    trimesh_body.visual.face_colors = [0,255,0,255]
    trimesh_bps = trimesh.points.PointCloud(bps * cube_size - shift, colors=[0, 255, 255, 100])
    trimesh_scene_crop = trimesh.points.PointCloud(scene_verts_crop_local - shift, colors=[255, 255, 0, 100])

    s = trimesh.Scene()
    s.add_geometry(trimesh_env)
    s.add_geometry(trimesh_body)
    s.add_geometry(trimesh_scene_crop)
    s.show(caption="Crop scene")


    s = trimesh.Scene()
    s.add_geometry(trimesh_env)
    s.add_geometry(trimesh_body)
    s.add_geometry(trimesh_bps)
    s.show(caption="Basis point set (randomly generated)")




    import random
    indxs = random.sample(range(0, 10000), 500)
    trimesh_bpsenv_reduced = trimesh.points.PointCloud(selected_scene_verts_global[indxs] * cube_size - shift, colors=[255, 255, 0, 100])
    rays_bps2env = np.hstack((selected_scene_verts_global[indxs] * cube_size - shift, bps[indxs] * cube_size - shift))
    trimesh_rays_bps2env = trimesh.load_path(rays_bps2env.reshape(-1, 2, 3))
    trimesh_rays_bps2env.colors = np.ones((len(trimesh_rays_bps2env.entities), 4)) * [0, 255, 255, 200]

    s = trimesh.Scene()
    s.add_geometry(trimesh_env)
    s.add_geometry(trimesh_body)
    s.add_geometry(trimesh_bpsenv_reduced)
    s.add_geometry(trimesh_rays_bps2env)
    s.show(caption="From basis point set to environment")

    rays_env2body = np.hstack((selected_scene_verts_global[indxs] * cube_size - shift, selected_body_verts_global[indxs] * cube_size - shift))
    trimesh_ray_env2body = trimesh.load_path(rays_env2body.reshape(-1, 2, 3))
    trimesh_ray_env2body.colors = np.ones((len(trimesh_ray_env2body.entities), 4)) * [255, 255, 0, 100]
    trimesh_bpsbody_reduced = trimesh.points.PointCloud(selected_body_verts_global[indxs] * cube_size - shift, colors=[0, 255, 0])

    s = trimesh.Scene()
    s.add_geometry(trimesh_env)
    s.add_geometry(trimesh_body)
    s.add_geometry(trimesh_bpsbody_reduced)
    s.add_geometry(trimesh_ray_env2body)
    s.show(caption="From env_bps to body")

    full_rays_env2body = np.hstack((selected_scene_verts_global * cube_size - shift, selected_body_verts_global * cube_size - shift))
    trimesh_full_rays_env2body = trimesh.load_path(full_rays_env2body.reshape(-1, 2, 3))
    trimesh_full_rays_env2body.colors = np.ones((len(trimesh_full_rays_env2body.entities), 4)) * [255, 255, 0, 100]

    s = trimesh.Scene()
    s.add_geometry(trimesh_env)
    s.add_geometry(trimesh_body)
    s.add_geometry(trimesh_full_rays_env2body)
    s.show(caption="FULL From env_bps to body")




    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
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


    body_verts = torch.tensor(body_verts_global[np.newaxis, :, :]).float().to(device)
    # body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
    # body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale
    # body_bps_sample = bps_encode_body(selected_scene_verts_global, normalized_body_verts)  # [n_feat, n_bps]
    body_bps = torch.from_numpy(body_bps[np.newaxis,:, :]).float().to(device)

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
        temp = body_verts_rec / cube_size  # scale into unit ball
        body_verts_rec_global = temp

        # calculate optimized body bps feature
        body_bps_rec = torch.zeros(bps.shape)
        if weight_loss_rec_bps > 0:
            nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
                body_verts_rec_global.detach().cpu().numpy())
            neigh_dist, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
            body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - torch.from_numpy(
                selected_scene_verts_global).float().to(device)  # [n_bps, 3]
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