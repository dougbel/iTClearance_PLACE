import trimesh
import os
import json
from  os.path import join as opj

from util.util_mesh import shift_rotate_vertices
from util.util_mesh import read_full_mesh

if __name__ == "__main__":

    datasets_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data/datasets/"
    to_explore = None

    # output_dir = "./output/MPH11"
    # Sit comfortable
    # to_explore = ['L_Leg_R_Leg20210514003056']
    # Sit normal
    # to_explore = [ 'L_Leg_R_Leg20210514003223', 'L_Leg_R_Leg20210514003306', 'L_Leg_R_Leg20210514004053', 'L_Leg_R_Leg20210514003350','L_Leg_R_Leg20210514004053']
    # #Stand up with arms down
    # to_explore = ['L_Leg_R_Leg20210514003720']
    # #Stand up with open arms
    # to_explore = ['L_Leg_R_Leg20210514003015']

    output_dir = "./output/N3Office"
    # #Reaching out mid up
    # to_explore = ['L_Leg_R_Leg20210518135205', 'L_Leg_R_Leg20210518135504', 'L_Leg_R_Leg20210518135718']
    # #Reaching out up
    # to_explore = ['L_Leg_R_Leg20210518135418']

    if to_explore is None:
        subdirs = os.listdir(output_dir)
    else:
        subdirs = to_explore

    subdirs.sort()
    last_env_name = None


    for l in subdirs:
        sub_d= opj(output_dir,l)
        with open(opj(sub_d, "data.json")) as f:
            d_dict=json.load(f)

        print("directory", l)
        print("eps", d_dict['eps'])


        trimesh_body_sample = trimesh.load(opj(sub_d, "body_sample.ply"))
        trimesh_body_s1 = trimesh.load(opj(sub_d, "body_s1.ply"))
        trimesh_body_s2 = trimesh.load(opj(sub_d, "body_s2.ply"))

        trimesh_body_sample.vertices = shift_rotate_vertices(trimesh_body_sample.vertices, d_dict["rotate_angle1"], d_dict["shift"])
        trimesh_body_sample.visual.face_colors=[200, 200, 200, 100]
        trimesh_body_s2.vertices = shift_rotate_vertices(trimesh_body_s2.vertices, d_dict["rotate_angle1"], d_dict["shift"])
        trimesh_body_s2.visual.face_colors = [200, 200, 200, 255]

        if last_env_name != d_dict['scene']:
            scene_trimesh = read_full_mesh(datasets_dir, d_dict['scene'])
            last_env_name = d_dict['scene']


        s = trimesh.Scene()
        s.add_geometry(scene_trimesh)
        s.add_geometry(trimesh_body_sample)
        s.add_geometry(trimesh_body_s2)
        s.show(caption=d_dict['scene']+"_"+l, resolution=(1024,768))
        # s.export(opj(output_dir, 'scene.ply'))
