import logging

import numpy as np
import trimesh
import vedo
from vedo import Plotter, write, merge
import os

from it_clearance.preprocessing.bubble_filler import BubbleFiller
from util.util_mesh import find_yaw_to_align_XY_OBB_with_BB

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    data_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/prox/scenes"

    output_dir = "output/filled_scenes_prox"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(data_dir):
        env_file = os.path.join(data_dir, file_name)

        trimesh_env = trimesh.load(env_file)
        rot_angle_1 = find_yaw_to_align_XY_OBB_with_BB(trimesh_env)
        vedo_env = vedo.load(env_file)
        vedo_env.rotateZ(rot_angle_1, rad=True)
        output_file = os.path.join(output_dir, file_name)
        vedo.write(vedo_env, output_file)


        filler = BubbleFiller(output_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        fine_bubbles.rotateZ(-rot_angle_1, rad=True)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        gross_bubbles.rotateZ(-rot_angle_1, rad=True)
        floor_filler = filler.calculate_floor_holes_filler(0.12)
        floor_filler.rotateZ(-rot_angle_1, rad=True)
        vedo_env.rotateZ(-rot_angle_1, rad=True)


        write(fine_bubbles, os.path.join(output_dir, "filler_fine_bubbles_"+file_name))
        write(gross_bubbles, os.path.join(output_dir, "filler_gross_bubbles_"+file_name))
        write(floor_filler, os.path.join(output_dir, "filler_floor_holes_"+file_name))
        write(merge(vedo_env, gross_bubbles, floor_filler, fine_bubbles), output_file)


    data_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/replica_v1/scenes"
    output_dir = os.path.join("./output/filled_scenes_replica_v1")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(data_dir):
        env_file = os.path.join(data_dir, file_name)

        trimesh_env = trimesh.load(env_file)
        rot_angle_1 = find_yaw_to_align_XY_OBB_with_BB(trimesh_env)
        vedo_env = vedo.load(env_file)
        vedo_env.rotateZ(rot_angle_1, rad=True)
        output_file = os.path.join(output_dir, file_name)
        vedo.write(vedo_env, output_file)

        filler = BubbleFiller(output_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        fine_bubbles.rotateZ(-rot_angle_1, rad=True)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        gross_bubbles.rotateZ(-rot_angle_1, rad=True)
        floor_filler = filler.calculate_floor_holes_filler(0.12)
        floor_filler.rotateZ(-rot_angle_1, rad=True)
        vedo_env.rotateZ(-rot_angle_1, rad=True)

        write(fine_bubbles, os.path.join(output_dir, "filler_fine_bubbles_" + file_name))
        write(gross_bubbles, os.path.join(output_dir, "filler_gross_bubbles_" + file_name))
        write(floor_filler, os.path.join(output_dir, "filler_floor_holes_" + file_name))
        write(merge(vedo_env, gross_bubbles, floor_filler, fine_bubbles), output_file)


    data_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/mp3d/scenes"
    output_dir = os.path.join("./output/filled_scenes_mp3d")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(data_dir):
        env_file = os.path.join(data_dir, file_name)

        trimesh_env = trimesh.load(env_file)
        rot_angle_1 = find_yaw_to_align_XY_OBB_with_BB(trimesh_env)
        vedo_env = vedo.load(env_file)
        vedo_env.rotateZ(rot_angle_1, rad=True)
        output_file = os.path.join(output_dir, file_name)
        vedo.write(vedo_env, output_file)

        filler = BubbleFiller(output_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        fine_bubbles.rotateZ(-rot_angle_1, rad=True)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        gross_bubbles.rotateZ(-rot_angle_1, rad=True)
        floor_filler = filler.calculate_floor_holes_filler(0.12)
        floor_filler.rotateZ(-rot_angle_1, rad=True)
        vedo_env.rotateZ(-rot_angle_1, rad=True)

        write(fine_bubbles, os.path.join(output_dir, "filler_fine_bubbles_" + file_name))
        write(gross_bubbles, os.path.join(output_dir, "filler_gross_bubbles_" + file_name))
        write(floor_filler, os.path.join(output_dir, "filler_floor_holes_" + file_name))
        write(merge(vedo_env, gross_bubbles, floor_filler, fine_bubbles), output_file)