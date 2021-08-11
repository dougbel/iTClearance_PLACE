import json
import math
import os
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import trimesh

from tqdm import tqdm
import chamfer_pytorch.dist_chamfer as ext

from prettytable import PrettyTable

from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from os.path import join as opj

from vedo import Lines, Spheres, load, colorMap, trimesh2vtk

from it_clearance.testing.tester import TesterClearance
from preprocess.bps_encoding import bps_gen_ball_inside, bps_encode_scene
from preprocess.preprocess_optimize import augmentation_crop_scene_smplx
from si.fulldataclearancescores import FullDataClearanceScores
from util.util_mesh import find_files_mesh_env, read_sdf
from util.util_preprocessing import crop_scene_cube_smplx_at_point
from util.util_proxd import load_smplx_model, load_vposer_model, translate_smplx_body, rotate_smplx_body, \
    get_vertices_from_body_params, get_trimesh_from_body_params, device
from utils import get_contact_id, convert_to_6D_rot, convert_to_3D_rot, gen_body_mesh
from view.ViewPointScorePROXD import ViewPointScorePROXD

warnings.simplefilter("ignore", UserWarning)

class ControlPointScorePROXD():
    visualized_samples = []
    scores_data = None
    view = None

    max_limit_score = None
    max_limit_missing = None
    max_limit_cv_collided = None

    np_points = None
    np_scores = None
    np_missings = None
    np_cv_vollided = None

    np_bad_normal_points = None

    np_full_points = None

    affordance_name = None
    tester = None

    smplx_model = None
    vposer_model = None
    body_gender =  None

    def __init__(self, trainings_dir, json_conf_execution_file,
                 env_test_results_dir, prop_configs_dir, smplx_model_dir, vposer_model_dir,
                 datasets_dir, dataset_name, env_test_name):


        self.tester = TesterClearance(trainings_dir, json_conf_execution_file)
        self.affordance_name = self.tester.affordances[0][0]
        affordance_object = self.tester.affordances[0][1]
        subdir_name = self.affordance_name + "_" + affordance_object
        env_test_results = opj(env_test_results_dir, subdir_name)
        self.file_mesh_env = opj(env_test_results, "test_environment.ply")

        propagation_settings_file = opj(prop_configs_dir, subdir_name, 'propagation_data.json')
        with open(propagation_settings_file) as json_file:
            propagation_settings = json.load(json_file)
        self.max_limit_score = propagation_settings['max_limit_score']
        self.max_limit_missing = propagation_settings['max_limit_missing']
        self.max_limit_cv_collided = propagation_settings['max_limit_cv_collided']

        df_scores_data = pd.read_csv(opj(env_test_results, "test_scores.csv"))
        self.scores_data = FullDataClearanceScores(df_scores_data, self.affordance_name)

        self.view = ViewPointScorePROXD(self, self.file_mesh_env)

        self.np_points, self.np_scores, self.np_missings, self.np_cv_collided = self.scores_data.filter_data_scores(
            self.max_limit_score,
            self.max_limit_missing,
            self.max_limit_cv_collided)
        self.view.add_point_cloud(self.np_points, self.np_scores, at=0)

        # draw sampled points with environment BAD normal
        self.np_bad_normal_points = self.scores_data.np_bad_normal_points
        scores = np.zeros(self.np_bad_normal_points.shape[0])
        scores.fill(self.max_limit_score)
        self.view.add_point_cloud(self.np_bad_normal_points, scores, at=0)

        self.np_full_points = np.concatenate((self.np_points, self.np_bad_normal_points), axis=0)
        self.np_full_scores = np.concatenate((self.np_scores, scores), axis=0)

        self.body_gender = self.tester.it_descriptor.definition["extra"]["body_gender"]
        self.contact_regions = self.tester.it_descriptor.definition["extra"]["contact_regions"]

        self.smplx_model = load_smplx_model(smplx_model_dir, self.body_gender)
        self.vposer_model = load_vposer_model(vposer_model_dir)

        body_params_file = self.tester.it_descriptor.object_filename()[:-11]+"_smplx_body_params.npy"
        self.np_body_params = np.load(body_params_file)

        self.datasets_dir = datasets_dir
        self.dataset_dir = opj(datasets_dir, dataset_name)
        self.prox_dataset_dir = opj(datasets_dir, "prox")
        self.env_test_name = env_test_name

        self.s_grid_min_batch, self.s_grid_max_batch, self.s_sdf_batch = read_sdf(self.dataset_dir, self.env_test_name)


    def start(self):
        self.view.show()

    def get_data_from_nearest_point_to(self, np_point):
        closest_index = distance.cdist([np_point], self.np_full_points).argmin()
        np_nearest_point = self.np_full_points[closest_index]
        np_nearest_score = self.np_full_scores[closest_index]

        best_angle=None

        self.view.add_point_cloud([np_nearest_point], [np_nearest_score], 20, at=0)

        df_point_data = self.scores_data.get_point_data(np_nearest_point)

        print("\nInteraction: ", self.affordance_name)
        str_point = "({:.4f}, {:.4f}, {:.4f})".format(np_nearest_point[0], np_nearest_point[1], np_nearest_point[2])
        print("Point: ", str_point)
        if closest_index >= self.np_points.shape[0]:
            table = PrettyTable()
            table.field_names = ["default_score", "info"]
            info = 'bad environment normal orientation'
            table.add_row([self.max_limit_score, info])
            print(table)

        else:
            table = PrettyTable()
            table.field_names = ["orientation", "cv collisions", "missing", "score", "info"]
            for row_index, row in df_point_data.iterrows():
                # precalculated data
                ori = int(row['orientation'])
                angle = row['angle']
                score = row['score']
                cv_collisions = int(row['cv_collided'])
                missings = int(row['missings'])

                if (self.np_scores[closest_index] == score
                        and self.np_missings[closest_index] == missings
                        and self.np_cv_collided[closest_index] == cv_collisions):
                    info = 'SELECTED'
                    best_angle = angle
                else:
                    info = ''
                table.add_row([ori, cv_collisions, missings, score, info])

                # provenance vectors
                idx_from = ori * self.tester.num_pv
                idx_to = idx_from + self.tester.num_pv
                pv_points = self.tester.compiled_pv_begin[idx_from:idx_to]
                pv_vectors = self.tester.compiled_pv_direction[idx_from:idx_to]
                # clearance vectors
                idx_from = ori * self.tester.num_cv
                idx_to = idx_from + self.tester.num_cv
                cv_points = self.tester.compiled_cv_begin[idx_from:idx_to]
                cv_vectors = self.tester.compiled_cv_direction[idx_from:idx_to]

                provenance_vectors = Lines(np_nearest_point+pv_points, np_nearest_point+pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
                clearance_vectors = Lines(np_nearest_point+cv_points, np_nearest_point+cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
                cv_from = Spheres(np_nearest_point+cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

                vedo_obj = load(self.tester.objs_filenames[0]).lighting("plastic")
                vedo_obj.c(colorMap(score, name='jet', vmin=0, vmax=self.max_limit_score))
                if cv_collisions > self.max_limit_cv_collided or missings > self.max_limit_missing:
                    if cv_collisions > self.max_limit_cv_collided:
                        vedo_obj.alpha(.05)
                        provenance_vectors.alpha(.05)
                        clearance_vectors.alpha(.05)
                        cv_from.alpha(.05)
                    if missings > self.max_limit_missing:
                        vedo_obj.c('black')
                        vedo_obj.alpha(.25)
                        provenance_vectors.alpha(.25)
                        clearance_vectors.alpha(.25)
                        cv_from.alpha(.25)
                else:
                    a = max([1 - cv_collisions * (1 / (self.max_limit_cv_collided + 1)), .20])
                    vedo_obj.alpha(a)
                    provenance_vectors.alpha(a)
                    clearance_vectors.alpha(a)
                    cv_from.alpha(a)

                vedo_obj.rotateZ(angle, rad=True)
                vedo_obj.pos(x=np_nearest_point[0], y=np_nearest_point[1], z=np_nearest_point[2])


                self.view.add_vedo_element(provenance_vectors, at=0)
                self.view.add_vedo_element(clearance_vectors, at=0)
                self.view.add_vedo_element(cv_from, at=0)
                self.view.add_vedo_element(vedo_obj, at=0)
            print(table)

        return np_nearest_point, best_angle


    def optimize_best_scored_position(self, np_point, best_angle):

        np_body_params = rotate_smplx_body(self.np_body_params, self.smplx_model, best_angle)
        np_body_params = translate_smplx_body(np_body_params, self.smplx_model, np_point)

        body_trimesh_proxd = get_trimesh_from_body_params(self.smplx_model, self.vposer_model, np_body_params)

        body_trimesh_proxd.visual.face_colors = [150, 150, 0, 255]
        body_vedo_proxd = trimesh2vtk(body_trimesh_proxd)
        self.view.add_vedo_element(body_vedo_proxd, at=1)

        selected_p = body_trimesh_proxd.bounding_box.centroid #body_trimesh_proxd.vertices.mean(axis=0)

        print('[INFO] Position selected.')

        ROTATE_CUBE = True
        cube_size = 2.0  # 3D cage size TODO: it could change to 2.5
        weight_loss_rec_verts = 1.0
        weight_loss_rec_bps = 1.0
        weight_loss_vposer = 0.02
        weight_loss_shape = 0.01
        weight_loss_hand = 0.01
        weight_collision = 8.0
        weight_loss_contact = 0.5
        itr_s2 = 100
        rot_angle_1 = 0  # TODO eliminate this parameter once tested

        scene_trimesh = trimesh.load(self.file_mesh_env)
        scene_verts = scene_trimesh.vertices


        scene_verts_local, scene_verts_crop_local, shift = crop_scene_cube_smplx_at_point(
            scene_verts, picked_point=selected_p, r=cube_size, with_wall_ceilling=True,
            random_seed=None,
            rotate=ROTATE_CUBE)

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

        ############################# load trained model ###############################
        # scene_bps = torch.from_numpy(scene_bps).float().unsqueeze(0).to(device)  # [1, 1, n_bps]
        # scene_bps_verts = torch.from_numpy(selected_scene_verts_local.transpose(1, 0)).float().unsqueeze(0).to(device)  # [1, 3, 10000]


        # position body params in the new reference point "shift"
        shifted_rotated_scene = trimesh.Trimesh(vertices=scene_verts_local, faces=scene_trimesh.faces)
        shifted_rotated_scene.visual.face_colors = scene_trimesh.visual.face_colors

        body_params = translate_smplx_body(np_body_params, self.smplx_model, shift)
        shifted_rotated_body_smplx_trimesh = get_trimesh_from_body_params(self.smplx_model, self.vposer_model, body_params)

        s = trimesh.Scene()
        s.add_geometry(shifted_rotated_scene)
        s.add_geometry(shifted_rotated_body_smplx_trimesh)
        s.add_geometry(trimesh.points.PointCloud(selected_scene_verts_local, colors=[0, 255, 0]))
        s.show(caption="smplx_model shifted and rotated", flags={'axis': True})

        # extract body params of body positioned at "shift" and scaled to cube size
        np_body_verts_sample = get_vertices_from_body_params(self.smplx_model, self.vposer_model, body_params) / cube_size
        body_verts_sample = torch.from_numpy(np_body_verts_sample).float().unsqueeze(0).to(device).permute(0, 2, 1)

        nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(np_body_verts_sample)
        np_body_bps_sample, neigh_ind = nbrs.kneighbors(selected_scene_verts_global)
        # np_body_bps_sample = np_body_verts_sample[neigh_ind.squeeze()] - selected_scene_verts_global
        # np_body_bps_sample = np.sqrt(np_body_bps_sample[:, 0] ** 2 + np_body_bps_sample[:, 1] ** 2 + np_body_bps_sample[:, 2] ** 2)
        body_bps_sample = torch.from_numpy(np_body_bps_sample).float().unsqueeze(0).unsqueeze(0).to(device)

        # body_trimesh_sampled = trimesh.Trimesh(np_body_verts_sample * cube_size, faces=self.smplx_model.faces)
        # body_trimesh_sampled.visual.face_colors = [0, 255, 255, 100]
        # s = trimesh.Scene()
        # s.add_geometry(shifted_rotated_scene)
        # s.add_geometry(body_trimesh_sampled)
        # s.show(caption="body sampled (verifying scale)", flags={'axis': True})



        vid, _ = get_contact_id(body_segments_folder=opj(self.prox_dataset_dir, 'body_segments'),
                                contact_body_parts=self.contact_regions)

        ################ stage 1 (simple optimization, without contact/collision loss) ######
        print('[INFO] start optimization stage 1...')
        body_params_rec = torch.from_numpy(body_params).float().unsqueeze(0).to(
            device)  # initialize smplx params, bs=1, local 3D cage coordinate system
        body_params_rec = convert_to_6D_rot(body_params_rec)
        body_params_rec.requires_grad = True




        print('[INFO] start optimization stage 2...')
        optimizer = optim.Adam([body_params_rec], lr=0.01)

        body_verts = body_verts_sample.permute(0, 2, 1)  # [1, 10475, 3]
        body_verts = body_verts * cube_size  # to local 3d cage coordinate system scale

        for step in tqdm(range(itr_s2)):
            optimizer.zero_grad()

            body_params_rec_72 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
            body_pose_joint = self.vposer_model.decode(body_params_rec_72[:, 16:48], output_type='aa').view(1, -1)
            body_verts_rec = gen_body_mesh(body_params_rec_72, body_pose_joint, self.smplx_model)[0]  # [n_body_vert, 3]

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
            norm_verts_batch = (body_verts_rec_prox - self.s_grid_min_batch) / (self.s_grid_max_batch - self.s_grid_min_batch) * 2 - 1
            n_verts = norm_verts_batch.shape[1]
            body_sdf_batch = F.grid_sample(self.s_sdf_batch.unsqueeze(1),
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

            # print("contact", loss_contact)
            # print("collision", loss_collision)

            loss = weight_loss_rec_verts * loss_rec_verts + weight_loss_rec_bps * loss_rec_bps + weight_loss_vposer * loss_vposer + weight_loss_shape * loss_shape + weight_loss_hand * loss_hand + weight_collision * loss_collision + weight_loss_contact * loss_contact
            loss.backward(retain_graph=True)
            optimizer.step()

        print('[INFO] optimization stage 2 finished.')

        body_params_opt_s2 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        np_body_params_optim = translate_smplx_body(body_params_opt_s2.detach().squeeze().cpu().numpy(), self.smplx_model, -shift)
        body_trimesh_optim = get_trimesh_from_body_params(self.smplx_model, self.vposer_model, np_body_params_optim)
        body_trimesh_optim.visual.face_colors = [255, 255, 255, 255]
        body_vedo_optim = trimesh2vtk(body_trimesh_optim)
        self.view.add_vedo_element(body_vedo_optim, at=1)

