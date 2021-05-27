import logging

import numpy as np
from vedo import Plotter, write, merge
import os

from it_clearance.preprocessing.bubble_filler import BubbleFiller


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    data_dir = "/media/apacheco/Tezcatlipoca1/datasets_place/prox/scenes"
    for file_name in os.listdir(data_dir):
        env_file = os.path.join(data_dir, file_name)

        filler = BubbleFiller(env_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        floor_filler = filler.calculate_floor_holes_filler(0.12)

        write(fine_bubbles, "./output/filled_scenes_prox/filler_fine_bubbles_"+file_name)
        write(gross_bubbles, "./output/filled_scenes_prox/filler_gross_bubbles_"+file_name)
        write(floor_filler, "./output/filled_scenes_prox/filler_floor_holes_"+file_name)
        write(merge(filler.vedo_env, gross_bubbles, floor_filler, ), "./output/filled_scenes_prox/"+file_name)


    data_dir = "/media/apacheco/Tezcatlipoca1/datasets_place/replica_v1_sample"
    for scene in os.listdir(data_dir):
        env_file = os.path.join(data_dir, scene, "mesh.ply")

        filler = BubbleFiller(env_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        floor_filler = filler.calculate_floor_holes_filler(0.12)

        sub_dir = os.path.join("./output/filled_scenes_replica_v1", scene)
        os.mkdir(sub_dir)
        write(fine_bubbles, os.path.join(sub_dir, "filler_fine_bubbles_" + scene+".ply"))
        write(gross_bubbles, os.path.join(sub_dir, "filler_gross_bubbles_" + scene+".ply"))
        write(floor_filler, os.path.join(sub_dir, "filler_floor_holes_" + scene+".ply"))
        write(merge(filler.vedo_env, gross_bubbles, floor_filler, ), os.path.join(sub_dir, scene+".ply"))


    data_dir = "/media/apacheco/Tezcatlipoca1/datasets_place/mp3d_sample"
    for file_name in os.listdir(data_dir):
        env_file = os.path.join(data_dir, file_name)

        filler = BubbleFiller(env_file)
        fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
        gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
        floor_filler = filler.calculate_floor_holes_filler(0.12)

        write(fine_bubbles, "./output/filled_scenes_mp3d/filler_fine_bubbles_"+file_name)
        write(gross_bubbles, "./output/filled_scenes_mp3d/filler_gross_bubbles_"+file_name)
        write(floor_filler, "./output/filled_scenes_mp3d/filler_floor_holes_"+file_name)
        write(merge(filler.vedo_env, gross_bubbles, floor_filler, ), "./output/filled_scenes_mp3d/"+file_name)