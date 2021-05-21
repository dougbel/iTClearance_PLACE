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
from models.optimizer import adjust_body_mesh_to_raw_guess
from qt_ui.seed_train_extractor import Ui_MainWindow
from util.util_mesh import remove_collision
from utils import convert_to_3D_rot, gen_body_mesh


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

        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.valueChanged[int].connect(self.changeValue)

        self.ui.chk_view_rgb.stateChanged.connect(self.change_view_rgb)

        # ### WORKING INDEXES
        self.datasets_dir = datasets_dir
        self.scene_dir = None
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

        vposer_model_path = f'{self.datasets_dir}/pretrained/body_models/vposer_v1_0'
        vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
        vposer_model = vposer_model.to(device)

        body_verts_sample = self.vedo_body.points()
        scene_verts = self.vedo_env.points()

        self.update_progressbar_detail(30, "Generating body mesh")
        body_params_rec, shift = adjust_body_mesh_to_raw_guess(body_verts_sample, scene_verts, vposer_model, self.smplx_model, vedo_env=self.vedo_env, body_faces= self.vedo_body.faces())
        body_params_opt_s1 = convert_to_3D_rot(body_params_rec)  # tensor, [bs=1, 72]
        body_pose_joint_s1 = vposer_model.decode(body_params_opt_s1[:, 16:48], output_type='aa').view(1, -1)
        body_verts_opt_s1 = gen_body_mesh(body_params_opt_s1, body_pose_joint_s1, self.smplx_model)[0]
        body_verts_opt_s1 = body_verts_opt_s1.detach().cpu().numpy()

        body_trimesh_s1 = trimesh.Trimesh(vertices=body_verts_opt_s1, faces=self.smplx_model.faces,
                                          face_colors=[200, 200, 200, 255])
        body_trimesh_s1.visual.face_colors = [200, 200, 200, 255]


        tri_mesh_env = vedo.vtk2trimesh(self.vedo_env)
        tri_mesh_obj = vedo.vtk2trimesh(self.vedo_body)
        # env_name = self.recording_name
        # obj_name = self.vedo_text.GetText(1)

        self.update_progressbar_detail(50, "Removing collision")

        # for now the only option I have is to translate the body to an upper position
        # remove_collision(tri_mesh_env, tri_mesh_obj)

        self.update_progressbar_detail(50, "Collision eliminated")
        s = trimesh.Scene()
        tri_mesh_env.visual.face_colors = [200, 200, 200, 250]
        tri_mesh_obj.visual.face_colors = [0, 250, 0, 100]
        s.add_geometry(tri_mesh_env)
        s.add_geometry(tri_mesh_obj)
        s.add_geometry(body_trimesh_s1.apply_translation(-shift))
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

        ibs_init_size_sampling = 400
        ibs_resamplings = 4
        sampler_rate_ibs_samples = 5
        sampler_rate_generated_random_numbers = 500

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

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ui.lbl_recording.width(),
                                        self.ui.lbl_recording.height())  # , Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
