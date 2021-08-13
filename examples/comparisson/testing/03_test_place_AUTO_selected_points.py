import argparse
import warnings
from os.path import join as opj
from shutil import copyfile
import gc

import numpy as np
import pandas as pd
import trimesh

from tqdm import tqdm
import torch.optim as optim

from util.util_interactive import Selector
from util.util_mesh import read_full_mesh_sdf, define_scene_boundary_on_the_fly
from util.util_preprocessing import crop_scene_cube_smplx_at_point

from human_body_prior.tools.model_loader import load_vposer

import smplx
import chamfer_pytorch.dist_chamfer as ext
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from util.utils_files import get_file_names_with_extension_in
from utils import *

warnings.simplefilter("ignore", UserWarning)


def shift_rotate_mesh(body_verts, body_faces, shift, rotation):
    new_verts = np.zeros(body_verts.shape)
    temp = body_verts - shift
    new_verts[:, 0] = temp[:, 0] * math.cos(-rotation) - temp[:, 1] * math.sin(-rotation)
    new_verts[:, 1] = temp[:, 0] * math.sin(-rotation) + temp[:, 1] * math.cos(-rotation)
    new_verts[:, 2] = temp[:, 2]

    return trimesh.Trimesh(vertices=new_verts, faces=body_faces, face_colors=[200, 200, 200, 255])


def execute_place_in_picked_point(data_dir, dataset_name, scene_name, np_point, visualize = True):
    # set optimization hype-parameters
    weight_loss_rec_verts = 1.0
    weight_loss_rec_bps = 1.0
    weight_loss_vposer = 0.02
    weight_loss_shape = 0.01
    weight_loss_hand = 0.01
    weight_collision = 8.0
    weight_loss_contact = 0.5
    itr_s1 = 0
    itr_s2 = 300

    cube_size = 2.0  # 3D cage size
    optimize = True  # optimize or not

    # smplx/vpose model path
    smplx_model_path = f'{data_dir}/pretrained_place/body_models/smpl'
    vposer_model_path = f'{data_dir}/pretrained_place/body_models/vposer_v1_0'

    # trained model path
    scene_bps_AE_path = f'{data_dir}/pretrained_place/aes/sceneBpsAE_last_model.pkl'
    cVAE_path = f'{data_dir}/pretrained_place/aes/cVAE_last_model.pkl'
    scene_verts_AE_path = f'{data_dir}/pretrained_place/aes/sceneBpsVertsAE_last_model.pkl'
    bodyDec_path = f'{data_dir}/pretrained_place/aes/body_dec_last_model.pkl'

    # ### 2. Load scene mesh, scene SDF, smplx model, vposer model

    # In[2]:
    dataset_path = opj(data_dir, "datasets", dataset_name)
    prox_dataset_path = opj(data_dir, "datasets", "prox")
    # read scen mesh/sdf
    # scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(dataset_path,'prox',scene_name)
    scene_trimesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_full_mesh_sdf(dataset_path,
                                                                                                         scene_name)

    if visualize:
        # vp = vedo.Plotter(bg="white", axes=2)
        # vp.show([vedo.Sphere(np_point, r=.2, c="blue", alpha=1).lighting("plastic"),
        #          vedo.utils.trimesh2vtk(scene_trimesh).lighting('ambient')])
        s = trimesh.Scene()
        s.add_geometry(scene_trimesh)
        point_trimesh = trimesh.primitives.Sphere(radius=0.1, center=np_point)
        point_trimesh.visual.face_colors = [0, 0, 255]
        s.add_geometry(point_trimesh)

        s.show(caption=scene_name)


    smplx_model = smplx.create(smplx_model_path, model_type='smplx',
                               gender='neutral', ext='npz',
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
    print('[INFO] smplx model loaded.')

    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)
    print('[INFO] vposer model loaded')

    # ### 3. random select an area in the scene, and compute bps encodings
    # random place a 3D cage with cube size of 2 inside the 3D scene, compute the scene bps encoding, body bps encoding

    # In[3]:

    # rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary('prox', scene_name)
    rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary_on_the_fly(scene_trimesh)
    print('[INFO] calculated a)rotation to parallel, scene boundary .')

    scene_verts = rotate_scene_smplx_predefine(cur_scene_verts, rot_angle=rot_angle_1)

    np_point_rotated = rotate_scene_smplx_predefine(np.expand_dims(np_point, axis=0), rot_angle=rot_angle_1)[0]
    print('[INFO] rotated scene mesh to parallel.')


    ROTATE_CUBE = True

    scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_at_point(
        scene_verts, picked_point=np_point_rotated, r=cube_size, with_wall_ceilling=True,
        random_seed=None, #np.random.randint(10000),
        rotate=ROTATE_CUBE)


    np_point_rotated_and_shifted = np_point_rotated + shift


    print('[INFO] scene mesh cropped and shifted.')

    scene_basis_set = bps_gen_ball_inside(n_bps=10000, random_seed=100)
    scene_verts_global, scene_verts_crop_global, rot_angle_2 = augmentation_crop_scene_smplx(
        scene_verts_local / cube_size,
        scene_verts_crop_local / cube_size,
        np.random.randint(10000))
    scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set,
                                                                            scene_verts_crop_global)  # [n_feat, n_bps]
    selected_scene_verts_local = scene_verts_crop_local[selected_ind]
    print('[INFO] bps encoding computed.')

    # ### 4. load trained checkpoints, and random generate a body inside the selected area

    # In[4]:

    ############################# load trained model ###############################
    scene_bps = torch.from_numpy(scene_bps).float().unsqueeze(0).to(device)  # [1, 1, n_bps]
    scene_bps_verts = torch.from_numpy(selected_scene_verts_local.transpose(1, 0)).float().unsqueeze(0).to(
        device)  # [1, 3, 10000]

    scene_bps_AE = BPSRecMLP(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(scene_bps_AE_path, map_location=lambda storage, loc: storage)
    scene_bps_AE.load_state_dict(weights)

    c_VAE = BPS_CVAE(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, eps_d=32).to(device)
    weights = torch.load(cVAE_path, map_location=lambda storage, loc: storage)
    c_VAE.load_state_dict(weights)

    scene_AE = Verts_AE(n_bps=10000, hsize1=1024, hsize2=512).to(device)
    weights = torch.load(scene_verts_AE_path, map_location=lambda storage, loc: storage)
    scene_AE.load_state_dict(weights)

    body_dec = Body_Dec_shift(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
                              body_param_dim=75, rec_goal='body_verts').to(device)
    weights = torch.load(bodyDec_path, map_location=lambda storage, loc: storage)
    body_dec.load_state_dict(weights)

    scene_bps_AE.eval()
    c_VAE.eval()
    scene_AE.eval()
    body_dec.eval()

    print('[INFO] pretrained weights loaded.')

    # SHOW_OUTPUTS = True

    if visualize:
        shifted_rotated_scene_trimesh = trimesh.Trimesh(vertices=scene_verts_local, faces=scene_trimesh.faces)
        shifted_rotated_scene_trimesh.visual.face_colors = scene_trimesh.visual.face_colors

        unselected_scene_verts_local = [scene_verts_crop_local[i] for i in range(len(scene_verts_crop_local)) if
                                        i not in selected_ind]

        point_rotated_and_shifted_trimesh = trimesh.primitives.Sphere(radius=0.1, center=np_point_rotated_and_shifted)
        point_rotated_and_shifted_trimesh.visual.face_colors = [0, 0, 255]
        selected_scene_verts_local_trimesh = trimesh.points.PointCloud(selected_scene_verts_local, colors=[0, 255, 0])
        unselected_scene_verts_local_trimesh = trimesh.points.PointCloud(unselected_scene_verts_local, colors=[255, 255, 0])

        s = trimesh.Scene()
        s.add_geometry(shifted_rotated_scene_trimesh)
        s.add_geometry(point_rotated_and_shifted_trimesh)
        s.add_geometry(selected_scene_verts_local_trimesh)
        # s.add_geometry(unselected_scene_verts_local_trimesh)

        s.show(caption=scene_name)

    ######################## random sample a body  ##########################
    scene_bps_verts = scene_bps_verts / cube_size

    _, scene_bps_feat = scene_bps_AE(scene_bps)
    _, scene_bps_verts_feat = scene_AE(scene_bps_verts)

    # [1, 1, 10000]
    body_bps_sample = c_VAE.sample(1, scene_bps_feat)
    # [1, 3, 10475], unit ball scale, local coordinate
    body_verts_sample, body_shift = body_dec(body_bps_sample, scene_bps_verts_feat)
    # [bs, 10475, 3]
    body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_sample.shape[0], 10475, 3])
    # [bs=1, 3, 10475]
    body_verts_sample = body_verts_sample + body_shift.permute(0, 2, 1)

    print('[INFO] a random body is generated.')

    if visualize:
        body_trimesh_sampled = trimesh.Trimesh(
            vertices=body_verts_sample.detach().cpu().numpy().squeeze().transpose() * cube_size, faces=smplx_model.faces,
            face_colors=[200, 200, 200, 255])

        s = trimesh.Scene()
        s.add_geometry(body_trimesh_sampled)
        s.add_geometry(shifted_rotated_scene_trimesh)
        s.add_geometry(point_rotated_and_shifted_trimesh)
        s.add_geometry(selected_scene_verts_local_trimesh)
        # s.add_geometry(unselected_scene_verts_local_trimesh)
        s.show(caption=scene_name)

    # ### 5. Optimization stage 1: perform simple optimization (without interaction-based losses)

    # In[5]:

    # load contact parts
    contact_part = ['L_Leg', 'R_Leg']
    vid, _ = get_contact_id(body_segments_folder=os.path.join(prox_dataset_path, 'body_segments'),
                            contact_body_parts=contact_part)

    ################ stage 1 (simple optimization, without contact/collision loss) ######
    print('[INFO] start optimization stage 1...')
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

    body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
    body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

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
        body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
        body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_2) - temp[:, 1] * math.sin(rot_angle_2)
        body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_2) + temp[:, 1] * math.cos(rot_angle_2)
        body_verts_rec_global[:, 2] = temp[:, 2]

        # calculate optimized body bps feature
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

        ### body bps feature reconstruct loss
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

        loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand
        loss.backward(retain_graph=True)
        optimizer.step()

    print('[INFO] optimization stage 1 finished.')

    body_params_opt_s1 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
    body_pose_joint_s1 = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1, -1)
    body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint_s1, smplx_model)[0]
    body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()

    if visualize:
        body_trimesh_s1 = trimesh.Trimesh(vertices=body_verts_opt_s1, faces=smplx_model.faces,
                                          face_colors=[200, 200, 200, 255])
        s = trimesh.Scene()
        s.add_geometry(body_trimesh_s1)
        s.add_geometry(shifted_rotated_scene_trimesh)
        s.add_geometry(point_rotated_and_shifted_trimesh)
        s.add_geometry(selected_scene_verts_local_trimesh)
        # s.add_geometry(unselected_scene_verts_local_trimesh)

        s.show(caption=scene_name)

    # ### 6. Optimization stage 2: perform advanced optimizatioin (interaction-based), with contact and collision loss

    # In[6]:

    print('[INFO] start optimization stage 2...')
    optimizer = optim.Adam([body_params_rec], lr=0.01)

    body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
    body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

    for step in tqdm(range(itr_s2)):
        optimizer.zero_grad()

        body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
        body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]

        # transform body verts to unit ball global coordinate
        temp = body_verts_rec / cube_size  # scale into unit ball
        body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
        body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_2) - temp[:, 1] * math.sin(rot_angle_2)
        body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_2) + temp[:, 1] * math.cos(rot_angle_2)
        body_verts_rec_global[:, 2] = temp[:, 2]

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
        body_verts_rec_prox = torch.zeros(body_verts_rec.shape).to(device)
        temp = body_verts_rec - torch.from_numpy(shift).float().to(device)
        body_verts_rec_prox[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - temp[:, 1] * math.sin(-rot_angle_1)
        body_verts_rec_prox[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + temp[:, 1] * math.cos(-rot_angle_1)
        body_verts_rec_prox[:, 2] = temp[:, 2]
        body_verts_rec_prox = body_verts_rec_prox.unsqueeze(0)  # tensor, [bs=1, 10475, 3]

        ### sdf collision loss
        if (s_grid_min_batch is None):
            loss_collision = torch.tensor(0.0, dtype=torch.float32).to(device)
        else:
            norm_verts_batch = (body_verts_rec_prox - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) * 2 - 1
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
        body_verts_contact = body_verts_rec.unsqueeze(0)[:, vid, :]  # [1,1121,3]
        dist_chamfer_contact = ext.chamferDist()
        # scene_verts: [bs=1, n_scene_verts, 3]
        scene_verts = torch.from_numpy(scene_verts_local).float().to(device).unsqueeze(0)  # [1,50000,3]
        contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(), scene_verts.contiguous())
        loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))

        loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand + weight_collision * loss_collision + weight_loss_contact * loss_contact
        loss.backward(retain_graph=True)
        optimizer.step()

    print('[INFO] optimization stage 2 finished.')

    # ### 7. Visualize the optimized body

    # In[ ]:

    # smplx params --> body mesh
    body_params_opt_s2 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
    body_pose_joint = vposer_model.decode(body_params_opt_s2[:, 16:48], output_type='aa').view(1, -1)
    body_verts_opt_s2 = gen_body_mesh(body_params_opt_s2, body_pose_joint, smplx_model)[0]
    body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()  # [n_body_vert, 3]

    if visualize:
        body_trimesh_s2 = trimesh.Trimesh(vertices=body_verts_opt_s2, faces=smplx_model.faces,
                                          face_colors=[200, 200, 200, 255])
        s = trimesh.Scene()
        s.add_geometry(body_trimesh_s2)
        s.add_geometry(shifted_rotated_scene_trimesh)
        s.add_geometry(point_rotated_and_shifted_trimesh)
        s.add_geometry(selected_scene_verts_local_trimesh)
        # s.add_geometry(unselected_scene_verts_local_trimesh)
        s.show(caption=scene_name)

    # transfrom the body verts to the PROX world coordinate system
    # body_verts_opt_prox_s2 = np.zeros(body_verts_opt_s2.shape)  # [10475, 3]
    # temp = body_verts_opt_s2 - shift
    # body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - temp[:, 1] * math.sin(-rot_angle_1)
    # body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + temp[:, 1] * math.cos(-rot_angle_1)
    # body_verts_opt_prox_s2[:, 2] = temp[:, 2]

    # body_trimesh_opt_s2 = trimesh.Trimesh(vertices=body_verts_opt_prox_s2, faces=smplx_model.faces,
    #                                       face_colors=[200, 200, 200, 255])
    body_verts = body_verts.detach().cpu().numpy().squeeze()
    body_trimesh_no_opt = shift_rotate_mesh(body_verts, smplx_model.faces, shift, rot_angle_1)
    body_trimesh_opt_s1 = shift_rotate_mesh(body_verts_opt_s1, smplx_model.faces, shift, rot_angle_1)
    body_trimesh_opt_s2 = shift_rotate_mesh(body_verts_opt_s2, smplx_model.faces, shift, rot_angle_1)

    body_trimesh_no_opt.visual.face_colors = [200, 200, 200, 80]
    body_trimesh_opt_s1.visual.face_colors = [200, 200, 200, 150]
    body_trimesh_opt_s2.visual.face_colors = [200, 200, 200, 255]

    if visualize:
        s = trimesh.Scene()
        s.add_geometry(scene_trimesh)
        s.add_geometry(body_trimesh_no_opt)
        s.add_geometry(body_trimesh_opt_s1)
        s.add_geometry(body_trimesh_opt_s2)
        s.add_geometry(trimesh.primitives.Sphere(radius=0.1, center=np_point))
        s.show(caption=scene_name)
    return body_trimesh_no_opt, body_trimesh_opt_s1, body_trimesh_opt_s2


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, help='Information directory (dataset, pretrained models, etc)')
opt = parser.parse_args()
print(opt)


if __name__ == '__main__':

    # [ 'reaching_out_mid_up', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_mid',
    # 'sitting_looking_to_right', 'sitting_compact', 'reachin_out_ontable_one_hand'
    # 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot']

    # interaction = 'reaching_out_mid_up'

    base_dir = opt.base_dir

    directory_datasets = opj(base_dir, "datasets")

    general_points_dir = opj(base_dir, 'test', 'sampled_it_clearance')
    output_dir = opj(base_dir, 'test', 'sampled_place_exec')

    follow_up_file = opj(base_dir,'test', 'follow_up_process.csv')
    current_follow_up_column = "place_auto_samples_extracted"
    previus_follow_up_column = "it_auto_samples"

    follow_up_data = pd.read_csv(follow_up_file, index_col=[0, 1, 2])
    if not current_follow_up_column in follow_up_data.columns:
        follow_up_data[current_follow_up_column] = False

    num_total_task = follow_up_data.index.size
    pending_tasks = list(follow_up_data[ (follow_up_data[current_follow_up_column] == False)
                                         &  (follow_up_data[previus_follow_up_column]==True)].index)
    num_pending_tasks = len(pending_tasks)
    num_completed_task = follow_up_data[ (follow_up_data[current_follow_up_column] == True)
                                         &  (follow_up_data[previus_follow_up_column]==True)].index.size

    print( 'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, num_pending_tasks))

    for dataset_name, scene_name, interaction in pending_tasks:
        points_dir = opj(general_points_dir, scene_name, interaction)
        output_subdir = opj(output_dir, scene_name, interaction)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for np_point_file_name in get_file_names_with_extension_in(points_dir, ".npy"):
            np_point =  np.load(opj(points_dir, np_point_file_name))
            n = np_point_file_name[np_point_file_name.find("_")+1:np_point_file_name.find(".")]
            mesh_orig, mesh_opt1, mesh_opt2 = execute_place_in_picked_point(base_dir, dataset_name, scene_name, np_point, visualize=False)
            mesh_orig.export(opj(output_subdir, f"body_{n}_orig.ply"))
            # mesh_opt1.export(opj(output_subdir, f"body_{n}_opt1.ply"))
            mesh_opt2.export(opj(output_subdir, f"body_{n}_opt2.ply"))

        num_completed_task += 1
        num_pending_tasks -= 1
        copyfile(follow_up_file, follow_up_file + "_backup")
        follow_up_data.at[(dataset_name, scene_name, interaction), current_follow_up_column] = True
        follow_up_data.to_csv(follow_up_file)
        print(f"UPDATE: total {num_total_task}, done {num_completed_task}, pendings {num_pending_tasks}")
        gc.collect()