#!/usr/bin/env python
# coding: utf-8

# ## PLACE: Proximity Learning of Articulation and Contact in 3D Environments

# Here is a quick demo to generate random bodies given a scene mesh and visualize the results. Three steps are included:

#########################################################################
# ABEL you can pick position to test, you can use any environment to test it is not limited to those presented in paper
#########################################################################

# * Use the pretrained C-VAE to random generate body meshes
# * Optimization stage 1: perform simple optimization (without interaction-based losses)
# * Optimization stage 2: perform advanced optimizatioin (interaction-based)

# ### 1. Load dependencies, set data/model paths, and set hype-parameters for optimization
# we use PROX dataset, scene 'N3OpenArea', and please set the smplx/vpose model paths according to your configuration.

# In[1]:
import math
import pickle
import warnings

import trimesh
import vedo.utils

from util.util_interactive import Selector
from util.util_mesh import read_full_mesh_sdf, define_scene_boundary_on_the_fly
from util.util_preprocessing import crop_scene_cube_smplx_at_point
from util.util_proxd import get_trimesh_from_body_params, rotate_smplx_body, translate_smplx_body, \
    get_vertices_from_body_params
from utils_read_data import define_scene_boundary

warnings.simplefilter("ignore", UserWarning)

# from open3d import JVisualizer
import torch.optim as optim
from tqdm import tqdm
from human_body_prior.tools.model_loader import load_vposer
import smplx
import chamfer_pytorch.dist_chamfer as ext
from models.cvae import *
from preprocess.preprocess_optimize import *
from preprocess.bps_encoding import *
from utils import *






data_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

prox_dataset_path = f'{data_dir}/datasets_raw/prox'
dataset_path = f'{data_dir}/datasets_raw/prox'



# scene_name = 'BasementSittingBooth'
# recording_name= "BasementSittingBooth_03452_01"
# num_frame= 0
# contact_part = ['L_Leg', 'R_Leg']

scene_name = 'N0Sofa'
recording_name= "N0Sofa_00034_02"
num_frame= 562
contact_part = ["back", 'gluteus', 'thighs'] # 'L_Leg', 'R_Leg', 'back', 'gluteus', 'L_Hand', 'R_Hand', 'thighs'

# scene_name = 'MPH11'
# recording_name= "MPH11_00034_01"
# num_frame= 881
# contact_part = ['L_Leg', 'R_Leg', 'R_Hand']

# scene_name = 'MPH112'
# recording_name= "MPH112_00150_01"
# num_frame= 1148
# contact_part = [ "back", 'gluteus', 'thighs']




# smplx/vpose model path
smplx_model_path = f'{data_dir}/pretrained_place/body_models/smpl'
vposer_model_path = f'{data_dir}/pretrained_place/body_models/vposer_v1_0'

# set optimization hype-parameters
weight_loss_rec_verts = 1.0
weight_loss_rec_bps = 3.0
weight_loss_vposer = 0.02
weight_loss_shape = 0.01
weight_loss_hand = 0.01
weight_collision = 8.0
weight_loss_contact = 0.5
itr_s1 = 200
itr_s2 = 100

cube_size = 2.0  # 3D cage size
optimize = True  # optimize or not

# trained model path
scene_bps_AE_path =  f'{data_dir}/pretrained_place/aes/sceneBpsAE_last_model.pkl'
cVAE_path = f'{data_dir}/pretrained_place/aes/cVAE_last_model.pkl'
scene_verts_AE_path = f'{data_dir}/pretrained_place/aes/sceneBpsVertsAE_last_model.pkl'
bodyDec_path = f'{data_dir}/pretrained_place/aes/body_dec_last_model.pkl'


# ### 2. Load scene mesh, scene SDF, smplx model, vposer model

# In[2]:


# read scen mesh/sdf
# scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(prox_dataset_path,'prox',scene_name)
scene_trimesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_full_mesh_sdf(dataset_path,
                                                                                                     scene_name)

female_subjects_ids = [162, 3452, 159, 3403]
subject_id = int(recording_name.split('_')[1])
if subject_id in female_subjects_ids:
    gender = 'female'
else:
    gender = 'male'

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
print('[INFO] smplx model loaded.')

vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
vposer_model = vposer_model.to(device)
print('[INFO] vposer model loaded')

json_scene_conf = os.path.join(dataset_path, 'cam2world', scene_name + '.json')
with open(json_scene_conf, 'r') as f:
    cam2world = np.array(json.load(f))

fitting_dir = os.path.join(dataset_path, "PROXD", recording_name, "results")
frames_file_names = sorted(os.listdir(fitting_dir))

img_name = frames_file_names[num_frame]
with open(os.path.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
    param = pickle.load(f, encoding='latin1')
torch_param = {}
for key in param.keys():
    torch_param[key] = torch.tensor(param[key]).to(device)
output = smplx_model(return_verts=True, **torch_param)
vertices = output.vertices.detach().cpu().numpy().squeeze()
if np.isnan(vertices).all():
    vertices = np.zeros(vertices.shape)
tri_mesh_body = trimesh.Trimesh(vertices=vertices, faces=smplx_model.faces, face_colors=[200, 200, 200, 255])
tri_mesh_body.apply_transform(cam2world)
body_params = np.concatenate((param['transl'][0], param['global_orient'][0], param['betas'][0],
                              param['pose_embedding'][0],
                              param['left_hand_pose'][0], param['right_hand_pose'][0]), axis=0)

body_params = update_globalRT_for_smplx(body_params, smplx_model, cam2world)

# shift = tri_mesh_body.vertices.mean(axis=0)
# z_rotation = math.pi/2
# body_params = translate_smplx_body(body_params,smplx_model, -shift)
# body_params = rotate_smplx_body(body_params, smplx_model, z_rotation)

s = trimesh.Scene()
s.add_geometry(scene_trimesh)
s.add_geometry(tri_mesh_body)
# s.add_geometry(trimesh.primitives.Sphere(center=shift, radius=.1))
s.add_geometry(get_trimesh_from_body_params(smplx_model, vposer_model, body_params))
s.show(caption=f"{scene_name} smplx_model readed from PROX (cam2world tranformed)", flags={'axis': True})



# ### 3. random select an area in the scene, and compute bps encodings
# random place a 3D cage with cube size of 2 inside the 3D scene, compute the scene bps encoding, body bps encoding

# In[3]:

# rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary('prox', scene_name)
rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y =define_scene_boundary_on_the_fly(scene_trimesh)
print('[INFO] calculated a)rotation to parallel, scene boundary .')


scene_verts = rotate_scene_smplx_predefine(cur_scene_verts, rot_angle=rot_angle_1)
print('[INFO] rotated scene mesh to parallel.')

rotated_scene = trimesh.Trimesh(vertices=scene_verts, faces=scene_trimesh.faces)
rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors
z_rotation = rot_angle_1
body_params = rotate_smplx_body(body_params, smplx_model, z_rotation)
s = trimesh.Scene()
s.add_geometry(rotated_scene)
# s.add_geometry(tri_mesh_body) # if you  uncomment it will notice that the environment has been rotated
s.add_geometry(get_trimesh_from_body_params(smplx_model, vposer_model, body_params))
s.show(caption=f"{scene_name} smplx_model rotated with the environment", flags={'axis': True})


# sel_gui = Selector(rotated_scene, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
# selected_p = sel_gui.select_point_to_test()

selected_p = get_trimesh_from_body_params(smplx_model, vposer_model, body_params).vertices.mean(axis=0)

print('[INFO] Position selected.')

ROTATE_CUBE = True


scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_at_point(
     scene_verts, picked_point=selected_p, r=cube_size, with_wall_ceilling=True, random_seed=np.random.randint(10000),
     rotate=ROTATE_CUBE)


print('[INFO] scene mesh cropped and shifted.')

scene_basis_set = bps_gen_ball_inside(n_bps=10000, random_seed=100)
scene_verts_global, scene_verts_crop_global, rot_angle_2 =     augmentation_crop_scene_smplx(scene_verts_local / cube_size,
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
scene_bps_verts = torch.from_numpy(selected_scene_verts_local.transpose(1, 0)).float().unsqueeze(0).to(device)  # [1, 3, 10000]

# scene_bps_AE = BPSRecMLP(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512).to(device)
# weights = torch.load(scene_bps_AE_path, map_location=lambda storage, loc: storage)
# scene_bps_AE.load_state_dict(weights)
#
# c_VAE = BPS_CVAE(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, eps_d=32).to(device)
# weights = torch.load(cVAE_path, map_location=lambda storage, loc: storage)
# c_VAE.load_state_dict(weights)
#
# scene_AE = Verts_AE(n_bps=10000, hsize1=1024, hsize2=512).to(device)
# weights = torch.load(scene_verts_AE_path, map_location=lambda storage, loc: storage)
# scene_AE.load_state_dict(weights)
#
# body_dec = Body_Dec_shift(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
#                           body_param_dim=75, rec_goal='body_verts').to(device)
# weights = torch.load(bodyDec_path, map_location=lambda storage, loc: storage)
# body_dec.load_state_dict(weights)
#
# scene_bps_AE.eval()
# c_VAE.eval()
# scene_AE.eval()
# body_dec.eval()


print('[INFO] pretrained weights loaded.')


SHOW_OUTPUTS = True

shifted_rotated_scene = trimesh.Trimesh(vertices=scene_verts_local, faces=scene_trimesh.faces)
shifted_rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors


body_params = translate_smplx_body(body_params,smplx_model, shift)
shifted_rotated_body_smplx_trimesh = get_trimesh_from_body_params(smplx_model, vposer_model, body_params)


s = trimesh.Scene()
s.add_geometry(shifted_rotated_scene)
s.add_geometry(shifted_rotated_body_smplx_trimesh)
s.add_geometry(trimesh.points.PointCloud(selected_scene_verts_local, colors=[0,255,0]))
s.show(caption=f"{scene_name} smplx_model shifted and rotated", flags={'axis': True})



######################## random sample a body  ##########################
scene_bps_verts = scene_bps_verts / cube_size
np_body_verts_sample = get_vertices_from_body_params(smplx_model, vposer_model, body_params) / cube_size
body_verts_sample = torch.from_numpy(np_body_verts_sample).float().unsqueeze(0).to(device).permute(0, 2, 1)

nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(np_body_verts_sample)
np_body_bps_sample, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
# np_body_bps_sample = np_body_verts_sample[neigh_ind.squeeze()] - selected_scene_verts_global
# np_body_bps_sample = np.sqrt(np_body_bps_sample[:, 0] ** 2 + np_body_bps_sample[:, 1] ** 2 + np_body_bps_sample[:, 2] ** 2)
body_bps_sample = torch.from_numpy(np_body_bps_sample).float().unsqueeze(0).unsqueeze(0).to(device)


# _, scene_bps_feat = scene_bps_AE(scene_bps)
# _, scene_bps_verts_feat = scene_AE(scene_bps_verts)

# [1, 1, 10000]
# body_bps_sample = c_VAE.sample(1, scene_bps_feat)
# [1, 3, 10475], unit ball scale, local coordinate
# body_verts_sample, body_shift = body_dec(body_bps_sample, scene_bps_verts_feat)
# [bs, 10475, 3]
# body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_sample.shape[0], 10475, 3])
# [bs=1, 3, 10475]
# body_verts_sample = body_verts_sample + body_shift.permute(0, 2, 1)

print('[INFO] a random body is generated.')


# body_trimesh_sampled = trimesh.Trimesh(vertices=body_verts_sample.detach().cpu().numpy().squeeze().transpose()*cube_size, faces=smplx_model.faces, face_colors=[200, 200, 200, 255])
body_trimesh_sampled = trimesh.Trimesh(np_body_verts_sample*cube_size, faces=smplx_model.faces)
body_trimesh_sampled.visual.face_colors=[0, 255, 255, 100]
s = trimesh.Scene()
s.add_geometry(shifted_rotated_scene)
s.add_geometry(body_trimesh_sampled)
s.show(caption=scene_name+" body sampled", flags={'axis': True})

# ### 5. Optimization stage 1: perform simple optimization (without interaction-based losses)

# In[5]:



vid, _ = get_contact_id(body_segments_folder=os.path.join(prox_dataset_path, 'body_segments'),
                        contact_body_parts=contact_part)

################ stage 1 (simple optimization, without contact/collision loss) ######
print('[INFO] start optimization stage 1...')
body_params_rec = torch.from_numpy(body_params).float().unsqueeze(0).to(device)  # initialize smplx params, bs=1, local 3D cage coordinate system
body_params_rec = convert_to_6D_rot(body_params_rec)
body_params_rec.requires_grad = True
# optimizer = optim.Adam([body_params_rec], lr=0.1)
#
# body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
# body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale
#
# for step in tqdm(range(itr_s1)):
#     if step > 100:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.01
#     if step > 300:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.001
#     optimizer.zero_grad()
#
#     body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
#     body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
#     body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, smplx_model)[0]  # [n_body_vert, 3]
#
#     # transform body verts to unit ball global coordinate system
#     temp = body_verts_rec / cube_size  # scale into unit ball
#     body_verts_rec_global = torch.zeros(body_verts_rec.shape).to(device)
#     body_verts_rec_global[:, 0] = temp[:, 0] * math.cos(rot_angle_2) - temp[:, 1] * math.sin(rot_angle_2)
#     body_verts_rec_global[:, 1] = temp[:, 0] * math.sin(rot_angle_2) + temp[:, 1] * math.cos(rot_angle_2)
#     body_verts_rec_global[:, 2] = temp[:, 2]
#
#     # calculate optimized body bps feature
#     body_bps_rec = torch.zeros(body_bps_sample.shape)
#     if weight_loss_rec_bps > 0:
#         nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(
#             body_verts_rec_global.detach().cpu().numpy())
#         neigh_dist, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
#         body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - torch.from_numpy(selected_scene_verts_global).float().to(device)  # [n_bps, 3]
#         body_bps_rec = torch.sqrt(
#             body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]
#
#     ### body bps feature reconstruct loss
#     loss_rec_verts = F.l1_loss(body_verts_rec.unsqueeze(0), body_verts)
#     loss_rec_bps = F.l1_loss(body_bps_sample, body_bps_rec)
#
#     ### vposer loss
#     body_params_rec_72 = convert_to_3D_rot(body_params_rec)
#     vposer_pose = body_params_rec_72[:, 16:48]
#     loss_vposer = torch.mean(vposer_pose ** 2)
#     ### shape prior loss
#     shape_params = body_params_rec_72[:, 6:16]
#     loss_shape = torch.mean(shape_params ** 2)
#     ### hand pose prior loss
#     hand_params = body_params_rec_72[:, 48:]
#     loss_hand = torch.mean(hand_params ** 2)
#
#     loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand
#     loss.backward(retain_graph=True)
#     optimizer.step()
#
# print('[INFO] optimization stage 1 finished.')
#
# body_params_opt_s1 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
# body_pose_joint_s1 = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1, -1)
# body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint_s1, smplx_model)[0]
# body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()
#
# body_trimesh_s1 = trimesh.Trimesh(vertices=body_verts_opt_s1, faces=smplx_model.faces, face_colors=[200, 200, 200, 255])
#
# s = trimesh.Scene()
# s.add_geometry(shifted_rotated_scene)
# s.add_geometry(body_trimesh_sampled)
# s.add_geometry(body_trimesh_s1)
#
# s.show(caption=f"{scene_name} after SIMPLE optimization", flags={'axis': True} )


print('[INFO] start optimization stage 2...')
optimizer = optim.Adam([body_params_rec], lr=0.01)

body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

for step in tqdm(range(itr_s2)):
    optimizer.zero_grad()

    body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
    body_pose_joint = vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1,-1)
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
        body_bps_rec = body_verts_rec_global[neigh_ind.squeeze()] - torch.from_numpy(selected_scene_verts_global).float().to(device)  # [n_bps, 3]
        body_bps_rec = torch.sqrt(
            body_bps_rec[:, 0] ** 2 + body_bps_rec[:, 1] ** 2 + body_bps_rec[:, 2] ** 2).unsqueeze(0).unsqueeze(0)  # [bs=1, 1, n_bps]

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
    contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(),scene_verts.contiguous())
    loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))

    print("contact",loss_contact)
    print("collision",loss_collision)

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
body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()   # [n_body_vert, 3]

body_trimesh_s2 = trimesh.Trimesh(vertices=body_verts_opt_s2, faces=smplx_model.faces, face_colors=[200, 200, 200, 255])

s = trimesh.Scene()
s.add_geometry(shifted_rotated_scene)
s.add_geometry(body_trimesh_sampled)
s.add_geometry(body_trimesh_s2)
s.show(caption=f"{scene_name} after ADVANCE optimization", flags={'axis': True} )






# transfrom the body verts to the PROX world coordinate system
body_verts_opt_prox_s2 = np.zeros(body_verts_opt_s2.shape)  # [10475, 3]
temp = body_verts_opt_s2 - shift
body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angle_1) - temp[:, 1] * math.sin(-rot_angle_1)
body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angle_1) + temp[:, 1] * math.cos(-rot_angle_1)
body_verts_opt_prox_s2[:, 2] = temp[:, 2]


body_trimesh_opt_s2 = trimesh.Trimesh(vertices=body_verts_opt_prox_s2, faces=smplx_model.faces, face_colors=[0, 200, 200, 255])
s = trimesh.Scene()
s.add_geometry(scene_trimesh)
s.add_geometry(body_trimesh_opt_s2)
s.add_geometry(tri_mesh_body)
s.show(caption=f"{scene_name} FINAL RESULT", flags={'axis': True})

