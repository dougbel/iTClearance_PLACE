#!/usr/bin/env python
# coding: utf-8

# ## PLACE: Proximity Learning of Articulation and Contact in 3D Environments 

# Here is a quick demo to generate random bodies given a scene mesh and visualize the results. Three steps are included:
#########################################################################
# ABEL you can pick position to test,
# and the pose of the body is given by the user, NORMALLY DETERMINED PREVIOUSLY
#########################################################################

# * Use the pretrained C-VAE to random generate body meshes
# * Optimization stage 1: perform simple optimization (without interaction-based losses)
# * Optimization stage 2: perform advanced optimization (interaction-based)

# ### 1. Load dependencies, set data/model paths, and set hype-parameters for optimization
# we use PROX dataset, scene 'N3OpenArea', and please set the smplx/vpose model paths according to your configuration.

import warnings
from datetime import datetime as dt
# In[1]:
from os.path import join as opj

import numpy as np
import trimesh

from models.controled_cvae import BPS_CVAE_Sampler
from util.util_mesh import shift_rotate_vertices
from util.util_interactive import Selector
from util.util_mesh import read_full_mesh_sdf, define_scene_boundary_on_the_fly
from util.util_preprocessing import crop_scene_cube_smplx_at_point

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

data_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data"

prox_dataset_path = f'{data_dir}/datasets_raw/prox'

# some test scenes
scene_name = 'MPH1Library'
# scene_name = 'N0SittingBooth'
# scene_name = 'MPH16'
# scene_name = 'N3OpenArea'

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
scene_bps_AE_path = f'{data_dir}/pretrained_place/aes/sceneBpsAE_last_model.pkl'
cVAE_path = f'{data_dir}/pretrained_place/aes/cVAE_last_model.pkl'
scene_verts_AE_path = f'{data_dir}/pretrained_place/aes/sceneBpsVertsAE_last_model.pkl'
bodyDec_path = f'{data_dir}/pretrained_place/aes/body_dec_last_model.pkl'

# ### 2. Load scene mesh, scene SDF, smplx model, vposer model

# In[2]:


# read scen mesh/sdf
# scene_mesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_mesh_sdf(prox_dataset_path,'prox',scene_name)
scene_trimesh, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch = read_full_mesh_sdf(prox_dataset_path,
                                                                                                     scene_name)
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
print('[INFO] rotated scene mesh to parallel.')

rotated_scene = trimesh.Trimesh(vertices=scene_verts, faces=scene_trimesh.faces)
rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors

sel_gui = Selector(rotated_scene, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
selected_p = sel_gui.select_point_to_test()

print('[INFO] Position selected.')

PAR_ROTATE_CUBE = True

scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_at_point(
    scene_verts, picked_point=selected_p, r=cube_size, with_wall_ceilling=True, random_seed=np.random.randint(10000),
    rotate=PAR_ROTATE_CUBE)

print('[INFO] scene mesh cropped and shifted.')

scene_basis_set = bps_gen_ball_inside(n_bps=10000, random_seed=100)
scene_verts_global, scene_verts_crop_global, rot_angle_2 = augmentation_crop_scene_smplx(scene_verts_local / cube_size,
                                                                                         scene_verts_crop_local / cube_size,
                                                                                         np.random.randint(10000))
scene_bps, selected_scene_verts_global, selected_ind = bps_encode_scene(scene_basis_set,
                                                                        scene_verts_crop_global)  # [n_feat, n_bps]
selected_scene_verts_local = scene_verts_crop_local[selected_ind]
print('[INFO] bps encoding computed.')

# ### 4. load trained checkpoints, and random generate a body inside the selected area

# In[4]:


############################# load trained model ###############################
# [1, 1, n_bps]
scene_bps = torch.from_numpy(scene_bps).float().unsqueeze(0).to(device)
# [1, 3, 10000]
scene_bps_verts = torch.from_numpy(selected_scene_verts_local.transpose(1, 0)).float().unsqueeze(0).to(device)

scene_bps_AE = BPSRecMLP(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512).to(device)
weights = torch.load(scene_bps_AE_path, map_location=lambda storage, loc: storage)
scene_bps_AE.load_state_dict(weights)

c_VAE_sampler = BPS_CVAE_Sampler(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, eps_d=32).to(device)
weights = torch.load(cVAE_path, map_location=lambda storage, loc: storage)
c_VAE_sampler.load_state_dict(weights)

scene_AE = Verts_AE(n_bps=10000, hsize1=1024, hsize2=512).to(device)
weights = torch.load(scene_verts_AE_path, map_location=lambda storage, loc: storage)
scene_AE.load_state_dict(weights)

body_dec = Body_Dec_shift(n_bps=10000, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
                          body_param_dim=75, rec_goal='body_verts').to(device)
weights = torch.load(bodyDec_path, map_location=lambda storage, loc: storage)
body_dec.load_state_dict(weights)

scene_bps_AE.eval()
c_VAE_sampler.eval()
scene_AE.eval()
body_dec.eval()

print('[INFO] pretrained weights loaded.')

shifted_rotated_scene = trimesh.Trimesh(vertices=scene_verts_local, faces=scene_trimesh.faces)
shifted_rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors

unselected_scene_verts_local = [scene_verts_crop_local[i] for i in range(len(scene_verts_crop_local)) if
                                i not in selected_ind]
# vp=vedo.Plotter(bg="white", axes=2)
# vp.show([vedo.Spheres(selected_scene_verts_local, r=.007, c="green", alpha=1).lighting("plastic"),
#          vedo.Spheres(unselected_scene_verts_local, r=.002, c="yellow", alpha=1).lighting("plastic"),
#                   vedo.utils.trimesh2vtk(shifted_rotated_scene).lighting('ambient')])
# ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1,1,1)

s = trimesh.Scene()
s.add_geometry(shifted_rotated_scene)
s.add_geometry(trimesh.points.PointCloud(selected_scene_verts_local, colors=[0, 255, 0]))
s.add_geometry(trimesh.points.PointCloud(unselected_scene_verts_local, colors=[255, 255, 0]))

s.show(caption=scene_name)

######################## random sample a body  ##########################
scene_bps_verts = scene_bps_verts / cube_size

_, scene_bps_feat = scene_bps_AE(scene_bps)
_, scene_bps_verts_feat = scene_AE(scene_bps_verts)


# load contact parts
# 'back' , 'gluteus', 'L_Hand', 'R_Hand', 'thighs'
contact_part = ['L_Leg', 'R_Leg']
body_segments_folder = opj(prox_dataset_path, 'body_segments')
vid, _ = get_contact_id(body_segments_folder=body_segments_folder, contact_body_parts=contact_part)


#sitting
# l_eps= [0.3211876451969147, 0.30291152000427246, 0.1917506754398346, -1.0801798105239868, 0.5288713574409485, 0.42544519901275635, 0.14429150521755219, 1.5890942811965942, 0.6887556910514832, -0.9856522679328918, -0.39569568634033203, -0.13832347095012665, 0.3833952248096466, -0.3807806074619293, -0.6894510388374329, -0.6966151595115662, -1.3502663373947144, -0.8825099468231201, 0.15727220475673676, -0.11211740225553513, 0.6192719340324402, -0.2709498703479767, -0.32777494192123413, 1.6682298183441162, 0.8331549167633057, 1.1735516786575317, 0.5049965381622314, -0.16900944709777832, -0.5560457110404968, -1.0440925359725952, -0.07316339015960693, 0.823982298374176]
#standing_up
# l_eps=[0.16956020891666412, 1.4085702896118164, 0.0072653465904295444, -1.4187829494476318, -0.9700753688812256, -0.010892998427152634, 1.9623650312423706, -1.9440350532531738, -0.6912352442741394, 1.240987777709961, 0.14470607042312622, -0.39878183603286743, 1.568782091140747, -0.3524121046066284, -0.5409720540046692, -1.7174192667007446, 0.7875590324401855, 0.445740282535553, -0.05451492220163345, 1.5267279148101807, 1.5781673192977905, 0.2699022591114044, -1.469357967376709, -0.3935515880584717, -2.627614736557007, -2.582261085510254, -2.6494381427764893, 0.22386780381202698, -0.28273454308509827, -0.935623288154602, 0.7151066064834595, -0.8994593620300293]
#stand up with open arms
# l_eps=[0.4733414351940155, -0.01961071416735649, 1.1716448068618774, 0.5620517730712891, -0.5726583003997803, 0.5483508706092834, -0.6054643392562866, -0.9876313209533691, 1.4459930658340454, -0.17240563035011292, 0.8412200808525085, -2.423884153366089, -0.5146135091781616, 0.6476659774780273, -1.3826407194137573, -2.5404279232025146, 0.5562962889671326, -0.2605315148830414, -1.6869593858718872, 0.9133750200271606, 1.081752061843872, -0.07212606072425842, -0.4763771593570709, -0.7769327759742737, -2.06022310256958, 0.6440153121948242, 0.7344487905502319, -0.8504639863967896, 0.9857348203659058, 1.0159989595413208, 0.8394792675971985, -0.4465622901916504]
#reaching out midup
# l_eps = [1.2944220304489136, 0.0864105075597763, -0.22662010788917542, -1.0316238403320312, 0.6759200692176819, -0.8528022170066833, 0.4411291480064392, -2.5672218799591064, -0.020628679543733597, -0.5529402494430542, 0.09983441978693008, -0.343007355928421, -3.003326654434204, -0.5688924193382263, 1.0448912382125854, -0.24047966301441193, -1.9232925176620483, 0.33397772908210754, 0.8320872187614441, 0.9343909621238708, -0.21160884201526642, -0.863167405128479, 1.172050952911377, -1.6214462518692017, 0.7031725645065308, 0.8367262482643127, 0.8290581107139587, -0.88414466381073, -0.15868723392486572, -0.8769300580024719, -1.2545982599258423, 0.04338790476322174]
#reaching out up
l_eps = [1.402968168258667, 2.2493715286254883, 0.5770577192306519, 1.9300870895385742, 0.3185293674468994, 0.780296266078949, 1.6509901285171509, 0.3705998659133911, -1.1425409317016602, -0.4879962205886841, 0.18852324783802032, 1.0481958389282227, 0.013094241730868816, -1.0591486692428589, 0.11459261924028397, -0.3021009564399719, 0.7233641743659973, 2.0356621742248535, 1.6529829502105713, 0.8079134821891785, -1.9374477863311768, -0.8663265705108643, 0.8204371929168701, -0.4149518609046936, 0.5740309953689575, 0.3294282853603363, -0.10406390577554703, 0.9014312028884888, 0.2942025065422058, -1.3144419193267822, 1.012931227684021, 0.44900405406951904]

body_bps_sample, eps = c_VAE_sampler.sample_fixed(l_eps, scene_bps_feat)


# [1, 3, 10475], unit ball scale, local coordinate
body_verts_sample, body_shift = body_dec(body_bps_sample, scene_bps_verts_feat)
# [bs, 10475, 3]
body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_sample.shape[0], 10475, 3])
# [bs=1, 3, 10475]
body_verts_sample = body_verts_sample + body_shift.permute(0, 2, 1)

print('[INFO] a random body is generated.')

####################################################################################################################
body_trimesh_sampled = trimesh.Trimesh(
    vertices=body_verts_sample.detach().cpu().numpy().squeeze().transpose() * cube_size,
    faces=smplx_model.faces,
    face_colors=[200, 200, 200, 255])

# ### 5. Optimization stage 1: perform simple optimization (without interaction-based losses)
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

body_trimesh_s1 = trimesh.Trimesh(vertices=body_verts_opt_s1, faces=smplx_model.faces,
                                  face_colors=[200, 200, 200, 255])

# s = trimesh.Scene()
# s.add_geometry(shifted_rotated_scene)
# s.add_geometry(body_trimesh_s1)
# s.show(caption=scene_name)

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

body_trimesh_s2 = trimesh.Trimesh(vertices=body_verts_opt_s2, faces=smplx_model.faces)
body_trimesh_s2.face_colors=[200, 200, 200, 255]

body_trimesh_sampled.vertices = shift_rotate_vertices(body_trimesh_sampled.vertices, rot_angle_1, shift)
body_trimesh_sampled.visual.face_colors = [200, 200, 200, 100]

body_trimesh_s2.vertices = shift_rotate_vertices(body_trimesh_s2.vertices, rot_angle_1, shift)
body_trimesh_s2.visual.face_colors = [200, 200, 200, 255]


s = trimesh.Scene()
s.add_geometry(scene_trimesh)
s.add_geometry(body_trimesh_sampled)
s.add_geometry(body_trimesh_s2)
s.show(caption=scene_name)
