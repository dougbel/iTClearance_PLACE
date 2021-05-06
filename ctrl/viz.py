import gc
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import smplx
import torch
import trimesh
import vedo
import vtk
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from transforms3d.affines import decompose
from vedo import Plotter, load, Points, Lines, Spheres

from qt_ui.seed_train_extractor import Ui_MainWindow


class CtrlPropagatorVisualizer:

    def __init__(self, datasets_dir, recording_name):

        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)

        self.ui.lbl_recording.setAutoFillBackground(False)

        self.available_interactions = []

        self.vp = Plotter(qtWidget=self.ui.vtk_widget, bg="white")
        self.vp.show([], axes=0)

        self.ui.lbl_recording.setHidden(True)

        # ### BUTTON SIGNALS
        self.ui.btn_previous.setEnabled(True)
        self.ui.btn_previous.clicked.connect(self.click_btn_previous)
        self.ui.btn_next.setEnabled(True)
        self.ui.btn_next.clicked.connect(self.click_btn_next)

        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.valueChanged[int].connect(self.changeValue)

        self.ui.chk_view_rgb.stateChanged.connect(self.change_view_rgb)

        # ### WORKING INDEXES
        self.datasets_dir = None
        self.scene_dir = None
        self.fitting_dir = None
        self.frames_dir = None
        self.frames_file_names = None

        self.recording_name = None
        self.cam2world = None
        self.model = None

        self.vedo_env = None

        self.BODY_N_VERTEX = 10475

        self.initialize(datasets_dir, recording_name)

        MainWindow.show()
        sys.exit(app.exec_())


    def initialize(self,  datasets_dir, recording_name):
        self.datasets_dir = datasets_dir
        self.recording_name = recording_name

        scene_name = self.recording_name.split("_")[0]
        model_folder = os.path.join(self.datasets_dir, "pretained/body_models/smpl")
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

        self.model = self.load_model(model_folder, gender)
        self.cam2world = self.load_cam2world(json_scene_conf)
        self.vedo_env = vedo.load(os.path.join(self.scene_dir, scene_name + '.ply')).lighting('ambient')

        self.ui.horizontalSlider.setMaximum(len(self.frames_file_names)-1)


        self.set_camera(self.cam2world)
        self.load_frame(0)


    def changeValue(self, pos):
        self.load_frame(pos)


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
            torch_param[key] = torch.tensor(param[key])
        output = self.model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        tri_mesh_body = trimesh.Trimesh(vertices=vertices, faces=self.model.faces, face_colors=[200, 200, 200, 255])
        tri_mesh_body.apply_transform(self.cam2world)
        vedo_body = vedo.trimesh2vtk(tri_mesh_body)


        vedo_text= vedo.Text2D(img_name, pos='bottom-right', c='white', bg='black', font='Arial', s=.8, alpha=1)
        self.vp.show(self.vedo_env, vedo_body,vedo_text,  interactive=False, resetcam=(num_frame == 0))

        self.ui.vtk_widget.Render()

        if not self.ui.chk_view_rgb.isChecked() :
            self.ui.lbl_recording.setHidden(True)
        else:
            self.ui.lbl_recording.setHidden(False)
            if os.path.isfile(os.path.join(self.frames_dir, img_name + '.jpg')):
                frame_file = os.path.join(self.frames_dir, img_name + '.jpg')
                color_img = cv2.imread(frame_file)
                color_img = cv2.flip(color_img, 1)
            else:
                color_img =cv2.imread("../data/stand_by.jpg")

            self.ui.lbl_recording.setPixmap(self.convert_cv_qt(color_img))



    def load_cam2world(self, json_scene_conf):
        with open(json_scene_conf, 'r') as f:
            trans = np.array(json.load(f))
        return trans


    def load_model(self, model_folder , gender):
        model = smplx.create(model_folder, model_type='smplx',
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
        return model

    def change_view_rgb(self):
        self.load_frame(self.ui.horizontalSlider.value())

    def click_btn_previous(self):
        pos =  self.ui.horizontalSlider.value()
        next_pos = max(pos-1, self.ui.horizontalSlider.minimum())
        self.ui.horizontalSlider.setValue(next_pos)
        self.load_frame(next_pos)


    def click_btn_next(self):
        pos =  self.ui.horizontalSlider.value()
        next_pos = min(pos+1, self.ui.horizontalSlider.maximum())
        self.ui.horizontalSlider.setValue(next_pos)
        self.load_frame(next_pos)


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ui.lbl_recording.width(), self.ui.lbl_recording.height())  # , Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
   ctrl = CtrlPropagatorVisualizer(datasets_dir="/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data",
                   recording_name="N3Library_03403_01")
