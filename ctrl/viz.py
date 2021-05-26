import gc
import json
import os
import pickle
import sys

import cv2
import numpy as np
import smplx
import torch
import trimesh
import vedo
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QInputDialog
from transforms3d.affines import decompose
from vedo import Plotter
from vedo.utils import flatten

from human_body_prior.tools.model_loader import load_vposer
from it import util
from it.training.ibs import IBSMesh
from it.training.maxdistancescalculator import MaxDistancesCalculator
from it.training.sampler import OnGivenPointCloudWeightedSampler
from it_clearance.training.agglomerator import AgglomeratorClearance
from it_clearance.training.sampler import PropagateNormalObjectPoissonDiscSamplerClearance
from it_clearance.training.saver import SaverClearance
from it_clearance.training.trainer import TrainerClearance
from it_clearance.utils import get_vtk_items_cv_pv
from models.optimizer import adjust_body_mesh_to_raw_guess, get_scaledShifted_bps_sets, optimization_stage_2
from qt_ui.seed_train_extractor import Ui_MainWindow
from util.util_mesh import remove_collision
from utils import convert_to_3D_rot, gen_body_mesh, get_contact_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CtrlPropagatorVisualizer:

    def __init__(self, datasets_dir):

        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)

        self.ui.lbl_recording.setAutoFillBackground(False)

        self.available_interactions = []

        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        self.vp.show([], axes=0)

        self.ui.lbl_recording.setHidden(True)
        self.progresbar_hidden(True)

        # ### BUTTON SIGNALS

        self.ui.l_quali.itemSelectionChanged.connect(self.update_visualized_recording_qual)
        self.ui.l_quanti.itemSelectionChanged.connect(self.update_visualized_recording_quan)

        self.ui.btn_previous.clicked.connect(self.click_btn_previous)
        self.ui.btn_next.clicked.connect(self.click_btn_next)
        self.ui.btn_train.clicked.connect(self.click_btn_train)
        self.ui.btn_avoid_collision.clicked.connect(self.click_btn_avoid_collision)
        self.ui.btn_view_place_optim.clicked.connect(self.click_view_place_optim)
        self.ui.btn_run_place_optim.clicked.connect(self.click_run_place_optim)

        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.valueChanged[int].connect(self.changeValue)

        self.ui.chk_view_rgb.stateChanged.connect(self.change_view_rgb)

        # ### WORKING INDEXES
        self.datasets_dir = datasets_dir
        self.scene_dir = None
        self.sdf_dir = None
        self.body_segments_dir = None
        self.fitting_dir = None
        self.frames_dir = None
        self.frames_file_names = None

        self.recording_name = None
        self.cam2world = None
        self.smplx_model = None

        self.vedo_env = None
        self.vedo_body = None
        self.vedo_text = None

        self.BODY_N_VERTEX = 10475
        self.CUBE_SIZE = 2

        self.list_prox_scans_qual()
        self.list_prox_scans_quan()

        MainWindow.show()
        sys.exit(app.exec_())

    def list_prox_scans_quan(self):
        l_dir = os.path.join(self.datasets_dir, "datasets/prox_quantitative", "fittings/mosh")
        list_scene = os.listdir(l_dir)
        list_scene.sort()
        for scan in list_scene:
            self.ui.l_quanti.addItem(scan)

    def list_prox_scans_qual(self):
        l_dir = os.path.join(self.datasets_dir, "datasets/prox", "PROXD")
        list_scene = os.listdir(l_dir)
        list_scene.sort()
        for scan in list_scene:
            self.ui.l_quali.addItem(scan)

    def update_progressbar_detail(self, value, message):
        self.ui.lbl_prg.setText(message)
        self.ui.prgbar_per.setValue(min(100, value))

    def progresbar_hidden(self, bool):
        self.ui.prgbar_per.setHidden(bool)
        self.ui.lbl_prg.setHidden(bool)

    def controls_enabled(self, bool):
        self.ui.btn_next.setEnabled(bool)
        self.ui.btn_previous.setEnabled(bool)
        self.ui.btn_train.setEnabled(bool)
        self.ui.horizontalSlider.setEnabled(bool)

    def update_visualized_recording_qual(self):
        if len(self.ui.l_quali.selectedItems()) > 0:
            self.controls_enabled(True)
            selection = self.ui.l_quali.selectedItems()[0].text()
            self.ui.l_quanti.clearSelection()
            self.initialize(selection)

    def update_visualized_recording_quan(self):
        if len(self.ui.l_quanti.selectedItems()) > 0:
            self.controls_enabled(True)
            selection = self.ui.l_quanti.selectedItems()[0].text()
            self.ui.l_quali.clearSelection()
            self.initialize(selection)

    def click_btn_train(self):
        gc.collect()
        text, ok = QInputDialog.getText(self.ui.centralwidget, 'Train interaction', 'Interaction name:')
        while text == "" and ok == True:
            text, ok = QInputDialog.getText(self.ui.centralwidget, 'Train interaction', 'Interaction name:')

        text = text.replace(" ", "_")

        if ok:
            self.progresbar_hidden(False)
            self.__train(text)

    def click_btn_avoid_collision(self):
        self.progresbar_hidden(False)
        self.update_progressbar_detail(5, "Initializing")
        tri_mesh_env = vedo.vtk2trimesh(self.vedo_env)
        tri_mesh_obj = vedo.vtk2trimesh(self.vedo_body)
        env_name = self.recording_name
        obj_name = self.vedo_text.GetText(1)

        self.update_progressbar_detail(50, "Removing collision")

        # for now the only option I have is to translate the body to an upper position
        remove_collision(tri_mesh_env, tri_mesh_obj)

        self.update_progressbar_detail(50, "Collision eliminated")
        s = trimesh.Scene()
        tri_mesh_env.visual.face_colors = [200, 200, 200, 250]
        tri_mesh_obj.visual.face_colors = [0, 250, 0, 255]
        s.add_geometry(tri_mesh_env)
        s.add_geometry(tri_mesh_obj)
        s.show(caption="Uncollided")
        self.progresbar_hidden(True)

    def click_view_place_optim(self):
        self.progresbar_hidden(False)
        self.update_progressbar_detail(5, "Initializing")

        np_body_verts_sample = self.vedo_body.points()
        np_body_faces = self.vedo_body.faces()
        np_scene_verts = self.vedo_env.points()
        cube_size = self.CUBE_SIZE

        scene_verts_crop_scaled, bps, scene_bps_feat, scene_bps_verts, body_bps_feat, body_bps_verts, shift = get_scaledShifted_bps_sets(
            np_body_verts_sample, np_scene_verts, cube_size)

        import trimesh
        import vedo
        trimesh_env = vedo.vtk2trimesh(self.vedo_env)
        trimesh_body = trimesh.Trimesh(np_body_verts_sample, np_body_faces)
        trimesh_body.visual.face_colors = [0, 255, 0, 255]
        trimesh_bps = trimesh.points.PointCloud(bps * cube_size - shift, colors=[0, 255, 255, 100])
        trimesh_scene_crop = trimesh.points.PointCloud(scene_verts_crop_scaled * cube_size - shift,
                                                       colors=[255, 255, 0, 100])


        self.update_progressbar_detail(20, "Showing Crop scene")
        s = trimesh.Scene()
        s.add_geometry(trimesh_env)
        s.add_geometry(trimesh_body)
        s.add_geometry(trimesh_scene_crop)
        s.show(caption="Crop scene", resolution=(1024,768))


        self.update_progressbar_detail(40, "Showing Basis point set (randomly generated)")
        s = trimesh.Scene()
        s.add_geometry(trimesh_env)
        s.add_geometry(trimesh_body)
        s.add_geometry(trimesh_bps)
        s.show(caption="Basis point set (randomly generated)", resolution=(1024,768))

        import random
        indxs = random.sample(range(0, 10000), 500)
        trimesh_bpsenv_reduced = trimesh.points.PointCloud(scene_bps_verts[indxs] * cube_size - shift,
                                                           colors=[255, 255, 0, 100])
        rays_bps2env = np.hstack((scene_bps_verts[indxs] * cube_size - shift, bps[indxs] * cube_size - shift))
        trimesh_rays_bps2env = trimesh.load_path(rays_bps2env.reshape(-1, 2, 3))
        trimesh_rays_bps2env.colors = np.ones((len(trimesh_rays_bps2env.entities), 4)) * [0, 255, 255, 200]

        self.update_progressbar_detail(60, "Showing From basis point set to environment")
        s = trimesh.Scene()
        s.add_geometry(trimesh_env)
        s.add_geometry(trimesh_body)
        s.add_geometry(trimesh_bpsenv_reduced)
        s.add_geometry(trimesh_rays_bps2env)
        s.show(caption="From basis point set to environment", resolution=(1024,768))

        rays_env2body = np.hstack(
            (scene_bps_verts[indxs] * cube_size - shift, body_bps_verts[indxs] * cube_size - shift))
        trimesh_ray_env2body = trimesh.load_path(rays_env2body.reshape(-1, 2, 3))
        trimesh_ray_env2body.colors = np.ones((len(trimesh_ray_env2body.entities), 4)) * [255, 255, 0, 100]
        trimesh_bpsbody_reduced = trimesh.points.PointCloud(body_bps_verts[indxs] * cube_size - shift,
                                                            colors=[0, 255, 0])

        self.update_progressbar_detail(80, "Showing From env_bps to body")
        s = trimesh.Scene()
        s.add_geometry(trimesh_env)
        s.add_geometry(trimesh_body)
        s.add_geometry(trimesh_bpsbody_reduced)
        s.add_geometry(trimesh_ray_env2body)
        s.show(caption="From env_bps to body", resolution=(1024,768))

        full_rays_env2body = np.hstack((scene_bps_verts * cube_size - shift, body_bps_verts * cube_size - shift))
        trimesh_full_rays_env2body = trimesh.load_path(full_rays_env2body.reshape(-1, 2, 3))
        trimesh_full_rays_env2body.colors = np.ones((len(trimesh_full_rays_env2body.entities), 4)) * [255, 255, 0, 100]

        self.update_progressbar_detail(95, "Showing FULL From env_bps to body")

        s = trimesh.Scene()
        s.add_geometry(trimesh_env)
        s.add_geometry(trimesh_body)
        s.add_geometry(trimesh_full_rays_env2body)
        s.show(caption="FULL From env_bps to body", resolution=(1024,768))


        self.progresbar_hidden(True)

    def click_run_place_optim(self):
        self.progresbar_hidden(False)
        self.update_progressbar_detail(5, "Initializing")
        vposer_model_path = f'{self.datasets_dir}/pretrained/body_models/vposer_v1_0'
        vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
        vposer_model = vposer_model.to(device)
        body_verts_sample = self.vedo_body.points()
        scene_verts = self.vedo_env.points()


        self.update_progressbar_detail(20, "Adjust body params to body mesh")
        body_params_rec, shift = adjust_body_mesh_to_raw_guess(body_verts_sample, scene_verts,
                                                            vposer_model, self.smplx_model,
                                                            weight_loss_rec_verts= self.ui.sbox_rec_vertices.value(), # 1.0
                                                            weight_loss_rec_bps = self.ui.sbox_rec_bps.value(), # 3.0
                                                            weight_loss_vposer = self.ui.sbox_v_poser.value(),  # 0.02
                                                            weight_loss_shape = self.ui.sbox_shape.value(),  # 0.01
                                                            weight_loss_hand = self.ui.sbox_hand.value(),  # 0.01
                                                            itr_s1 = self.ui.sbox_its1.value()  # 200
                                                            )


        self.update_progressbar_detail(50, "Transforming body params to mesh")
        body_params_opt_s1 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint_s1 = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint_s1, self.smplx_model)[0]
        body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()
        body_trimesh_s1 = trimesh.Trimesh(vertices=body_verts_opt_s1, faces=self.smplx_model.faces)
        body_trimesh_s1.visual.face_colors = [255, 161, 53, 200]


        self.update_progressbar_detail(80, "Adjusting body mesh with collision losses")
        s_grid_min_batch, s_grid_max_batch, s_sdf_batch = self.load_sdf()
        contact_part = self.get_cofigured_contact_regions() # 'L_Leg', 'R_Leg', 'back', 'gluteus', 'L_Hand', 'R_Hand', 'thighs'
        id_contact_vertices, _ = get_contact_id(self.body_segments_dir, contact_part)
        body_params_rec, shift = optimization_stage_2(body_verts_sample, scene_verts, vposer_model, self.smplx_model,
                                                      body_params_rec, s_grid_min_batch, s_grid_max_batch, s_sdf_batch,
                                                      id_contact_vertices,
                                                      weight_loss_rec_verts = self.ui.sbox_rec_vertices.value(), # 1.0
                                                      weight_loss_rec_bps = self.ui.sbox_rec_bps.value(), # 3.0
                                                      weight_loss_vposer = self.ui.sbox_v_poser.value(), # 0.02
                                                      weight_loss_shape = self.ui.sbox_shape.value(), # 0.01
                                                      weight_loss_hand = self.ui.sbox_hand.value(), # 0.01
                                                      weight_collision = self.ui.sbox_collision.value(), # 8.0
                                                      weight_loss_contact = self.ui.sbox_contact.value(), # 0.5
                                                      itr_s2 = self.ui.sbox_its2.value() # 100
                                                      )


        self.update_progressbar_detail(90, "Transforming body params to mesh")
        body_params_opt_s2 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint_s2 = vposer_model.decode(body_params_opt_s2[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s2 = gen_body_mesh(body_params_opt_s2, body_pose_joint_s2, self.smplx_model)[0]
        body_verts_opt_s2 = body_verts_opt_s2.detach().cpu().numpy()
        body_trimesh_s2 = trimesh.Trimesh(vertices=body_verts_opt_s2, faces=self.smplx_model.faces)
        body_trimesh_s2.visual.face_colors = [255, 0, 0, 255]

        self.update_progressbar_detail(95, "Parameters adjusted to body mesh")
        tri_mesh_env = vedo.vtk2trimesh(self.vedo_env)
        tri_mesh_obj = vedo.vtk2trimesh(self.vedo_body)
        s = trimesh.Scene()
        tri_mesh_env.visual.face_colors = [200, 200, 200, 250]
        tri_mesh_obj.visual.face_colors = [0, 250, 0, 100]
        s.add_geometry(tri_mesh_env)
        s.add_geometry(tri_mesh_obj)
        s.add_geometry(body_trimesh_s1.apply_translation(-shift))
        s.add_geometry(body_trimesh_s2.apply_translation(-shift))
        s.show(caption="Uncollided")
        self.progresbar_hidden(True)

    def __train(self, affordance_name):

        self.update_progressbar_detail(5, "Initializing")
        tri_mesh_env = vedo.vtk2trimesh(self.vedo_env)
        tri_mesh_obj = vedo.vtk2trimesh(self.vedo_body)
        affordance_name = affordance_name
        env_name = self.recording_name
        obj_name = self.vedo_text.GetText(1)

        self.update_progressbar_detail(10, "Removing collision")

        # for now the only option I have is to translate the body to an upper position
        remove_collision(tri_mesh_env, tri_mesh_obj)

        s = trimesh.Scene()
        tri_mesh_env.visual.face_colors = [200, 200, 200, 250]
        tri_mesh_obj.visual.face_colors = [0, 250, 0, 255]
        s.add_geometry(tri_mesh_env)
        s.add_geometry(tri_mesh_obj)
        s.show(caption="Uncollided")

        influence_radio_bb = 2
        extension, middle_point = util.influence_sphere(tri_mesh_obj, influence_radio_bb)

        tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

        ibs_init_size_sampling = self.ui.sbox_ibs_init_sampling.value()#400
        ibs_resamplings = self.ui.sbox_ibs_resampling.value()#4
        sampler_rate_ibs_samples = 5
        sampler_rate_generated_random_numbers = self.ui.sbox_pv_rate_rolls.value()

        ################################
        # GENERATING AND SEGMENTING IBS MESH
        ################################

        self.update_progressbar_detail(40, "Calculating IBS")

        influence_radio_ratio = 1.2
        ibs_calculator = IBSMesh(ibs_init_size_sampling, ibs_resamplings)
        ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)

        tri_mesh_ibs = ibs_calculator.get_trimesh()

        sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, influence_radio_ratio)
        tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

        np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]

        ################################
        # SAMPLING IBS MESH
        ################################

        self.update_progressbar_detail(70, "Training iT with Clearance Vectors")

        pv_sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                                      rate_generated_random_numbers=sampler_rate_generated_random_numbers)

        cv_sampler = PropagateNormalObjectPoissonDiscSamplerClearance()
        trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env,
                                   tri_mesh_obj=tri_mesh_obj, pv_sampler=pv_sampler, cv_sampler=cv_sampler)

        agglomerator = AgglomeratorClearance(trainer, num_orientations=8)

        max_distances = MaxDistancesCalculator(pv_points=trainer.pv_points, pv_vectors=trainer.pv_vectors,
                                               tri_mesh_obj=tri_mesh_obj, consider_collision_with_object=True,
                                               radio_ratio=influence_radio_ratio)

        output_subdir = "IBSMesh_" + str(ibs_init_size_sampling) + "_" + str(ibs_resamplings) + "_"
        output_subdir += pv_sampler.__class__.__name__ + "_" + str(sampler_rate_ibs_samples) + "_"
        output_subdir += str(sampler_rate_generated_random_numbers) + "_"
        output_subdir += cv_sampler.__class__.__name__ + "_" + str(cv_sampler.sample_size)

        self.update_progressbar_detail(90, "Saving")
        SaverClearance(affordance_name, env_name, obj_name, agglomerator,
                       max_distances, ibs_calculator, tri_mesh_obj, output_subdir)

        vedo_items = get_vtk_items_cv_pv(trainer.pv_points, trainer.pv_vectors, trainer.cv_points, trainer.cv_vectors,
                                         trimesh_obj=tri_mesh_obj,
                                         trimesh_ibs=tri_mesh_ibs_segmented)

        self.ui.vtk_widget.Render()
        self.vp.show(flatten([vedo_items, self.vedo_env, self.vedo_text]), interactive=False, resetcam=False)

        self.update_progressbar_detail(100, "Done")

    def initialize(self, recording_name):
        self.recording_name = recording_name

        scene_name = self.recording_name.split("_")[0]
        model_folder = os.path.join(self.datasets_dir, "pretrained/body_models/smpl")
        gender = "male"  # "female", "neutral"
        base_dir = os.path.join(self.datasets_dir, "datasets/prox_quantitative")
        pseudo_fitting_dir = os.path.join(base_dir, "fittings/mosh", self.recording_name)

        if os.path.isdir(pseudo_fitting_dir):
            json_scene_conf = os.path.join(base_dir, 'vicon2scene.json')
        else:
            base_dir = os.path.join(self.datasets_dir, "datasets/prox")
            female_subjects_ids = [162, 3452, 159, 3403]
            subject_id = int(self.recording_name.split('_')[1])
            if subject_id in female_subjects_ids:
                gender = 'female'
            else:
                gender = 'male'

            json_scene_conf = os.path.join(base_dir, 'cam2world', scene_name + '.json')
            pseudo_fitting_dir = os.path.join(base_dir, "PROXD", self.recording_name)

        self.fitting_dir = os.path.join(pseudo_fitting_dir, 'results')
        self.scene_dir = os.path.join(base_dir, 'scenes')
        self.sdf_dir =  os.path.join(base_dir, 'sdf')
        self.body_segments_dir =  os.path.join(base_dir, 'body_segments')
        self.frames_dir = os.path.join(base_dir, 'recordings', self.recording_name, 'Color')
        self.frames_file_names = sorted(os.listdir(self.fitting_dir))

        self.smplx_model = self.load_smplx_model(model_folder, gender)
        self.smplx_model.to(device)
        self.cam2world = self.load_cam2world(json_scene_conf)
        self.vedo_env = vedo.load(os.path.join(self.scene_dir, scene_name + '.ply')).lighting('ambient')

        self.ui.horizontalSlider.setMaximum(len(self.frames_file_names) - 1)

        self.set_camera(self.cam2world)
        self.ui.horizontalSlider.setValue(0)
        self.changeValue(0)

    def changeValue(self, pos):
        # loop = QEventLoop()
        # print(f"init pos {pos}")
        self.progresbar_hidden(True)
        self.load_frame(pos)
        # print(f"finish pos {pos}")
        # loop.exec_()

    def set_camera(self, cam2world):
        T, R, Z, S = decompose(cam2world)
        self.vp.camera.SetPosition(T)
        self.vp.camera.SetViewUp(np.matmul(R, [0, -1, 0]))

    def load_frame(self, num_frame):
        img_name = self.frames_file_names[num_frame]
        with open(os.path.join(self.fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f, encoding='latin1')
        torch_param = {}
        for key in param.keys():
            torch_param[key] = torch.tensor(param[key]).to(device)
        output = self.smplx_model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        if np.isnan(vertices).all():
            vertices = np.zeros(vertices.shape)
        tri_mesh_body = trimesh.Trimesh(vertices=vertices, faces=self.smplx_model.faces, face_colors=[200, 200, 200, 255])
        tri_mesh_body.apply_transform(self.cam2world)
        self.vedo_body = vedo.trimesh2vtk(tri_mesh_body)

        self.vedo_text = vedo.Text2D(img_name, pos='bottom-right', c='white', bg='black', font='Arial', s=.8, alpha=1)
        self.ui.vtk_widget.Render()
        self.vp.show(self.vedo_env, self.vedo_body, self.vedo_text, interactive=False, resetcam=(num_frame == 0))

        if not self.ui.chk_view_rgb.isChecked():
            self.ui.lbl_recording.setHidden(True)
        else:
            self.ui.lbl_recording.setHidden(False)
            if os.path.isfile(os.path.join(self.frames_dir, img_name + '.jpg')):
                frame_file = os.path.join(self.frames_dir, img_name + '.jpg')
                color_img = cv2.imread(frame_file)
                color_img = cv2.flip(color_img, 1)
            else:
                color_img = cv2.imread("data/stand_by.jpg")

            self.ui.lbl_recording.setPixmap(self.convert_cv_qt(color_img))

    def load_cam2world(self, json_scene_conf):
        with open(json_scene_conf, 'r') as f:
            trans = np.array(json.load(f))
        return trans

    def load_smplx_model(self, model_folder, gender):
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
        return smplx_model

    def load_sdf(self):
        ## read scene sdf
        scene_name = self.recording_name.split("_")[0]
        with open(os.path.join(self.sdf_dir, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(self.sdf_dir, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)  # [1, 256, 256, 256]

        return s_grid_min_batch, s_grid_max_batch, s_sdf_batch

    def change_view_rgb(self):
        self.load_frame(self.ui.horizontalSlider.value())

    def click_btn_previous(self):
        pos = self.ui.horizontalSlider.value()
        next_pos = max(pos - 1, self.ui.horizontalSlider.minimum())
        self.ui.horizontalSlider.setValue(next_pos)
        self.load_frame(next_pos)

    def click_btn_next(self):
        pos = self.ui.horizontalSlider.value()
        next_pos = min(pos + 1, self.ui.horizontalSlider.maximum())
        self.ui.horizontalSlider.setValue(next_pos)
        self.load_frame(next_pos)

    def get_cofigured_contact_regions(self):
        configured_regions=[]
        if(self.ui.chk_hand_left):
            configured_regions.append('L_Hand')
        if(self.ui.chk_hand_right):
            configured_regions.append('R_Hand')
        if(self.ui.chk_foot_left):
            configured_regions.append('L_Leg')
        if(self.ui.chk_foot_right):
            configured_regions.append('R_Leg')
        if(self.ui.chk_back):
            configured_regions.append('back')
        if(self.ui.chk_gluteus):
            configured_regions.append('gluteus')
        if(self.ui.chk_thighs):
            configured_regions.append('thighs')
        return configured_regions




    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ui.lbl_recording.width(),
                                        self.ui.lbl_recording.height())  # , Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
