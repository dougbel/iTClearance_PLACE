import os
import os.path as osp
import sys

import cv2
import numpy as np
import json
import open3d as o3d
import argparse

import torch
import pickle
import smplx
import trimesh
import vedo
import vtk
from PyQt5 import QtWidgets, uic
from transforms3d.affines import decompose
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

if __name__ == '__main__':

    data_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data"
    recording_name = "N3Library_03403_01" #N3Library_03403_02    vicon_03301_13
    scene_name = recording_name.split("_")[0]

    model_folder = osp.join(data_dir, "pretained/body_models/smpl")
    gender = "male"  # "female", "neutral"

    base_dir = osp.join(data_dir, "datasets/prox_quantitative")
    fitting_dir = osp.join(base_dir, "fittings/mosh",recording_name)

    if os.path.isdir( fitting_dir):
        json_scene_conf = os.path.join(base_dir, 'vicon2scene.json')
    else:
        base_dir = osp.join(data_dir, "datasets/prox")
        fitting_dir = osp.join(base_dir, "PROXD", recording_name)
        female_subjects_ids = [162, 3452, 159, 3403]
        subject_id = int(recording_name.split('_')[1])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male'
        json_scene_conf = os.path.join(base_dir, 'cam2world', scene_name + '.json')


    fitting_dir = osp.join(fitting_dir, 'results')
    scene_dir = osp.join(base_dir, 'scenes')
    recording_dir = osp.join(base_dir, 'recordings', recording_name)
    color_dir = os.path.join(recording_dir, 'Color')

    frame_files=sorted(os.listdir(fitting_dir))

    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)


    scene = o3d.io.read_triangle_mesh(osp.join(scene_dir, scene_name + '.ply'))
    with open(json_scene_conf, 'r') as f:
        trans = np.array(json.load(f))

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


    vp = vedo.Plotter( bg="white", axes=2)
    # vp = vedo.Plotter(qtWidget=vtkWidget, bg="white", axes=2)
    vedo_env = vp.load(osp.join(scene_dir, scene_name + '.ply')).lighting('ambient')
    T, R, Z, S = decompose(trans)
    vp.camera.SetPosition(T)
    vp.camera.SetViewUp(np.matmul(R, [0, -1, 0]))
    vp.addSlider2D(None, 0, len(frame_files), value=0, pos=[(.20, .04), (.8, .04)], title="Max missing")




    count = 0
    for img_name in frame_files[0::1]:

        with open(osp.join(fitting_dir, img_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f, encoding='latin1')

        torch_param = {}
        for key in param.keys():
            torch_param[key] = torch.tensor(param[key])
        output = model(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        tri_mesh_body = trimesh.Trimesh(vertices=vertices, faces=model.faces, face_colors=[200, 200, 200, 255])
        tri_mesh_body.apply_transform(trans)
        vedo_body =vedo.trimesh2vtk(tri_mesh_body)

        vp.show(vedo_env, vedo_body, interactive=False, resetcam=count==0)

        if os.path.isfile(os.path.join(color_dir, img_name + '.jpg')):
            frame_file = os.path.join(color_dir, img_name + '.jpg')
            color_img = cv2.imread(frame_file)
            color_img = cv2.flip(color_img, 1)
            cv2.imshow('frame', color_img)

        # key = cv2.waitKey(30)

        count += 1
