import json
import math

import trimesh
import numpy as np
import open3d as o3d
import torch
import os


def angle_between_2D_vectors(orig_2dvetor, dest_2dvector):
    """
    Return the angle to rotate to align the vector with the y axis
    :param vector_2d: the vector for measure the angle
    :return: positive or negative radians with respect to rotation
        >>> angle_between_2D_vectors((1, 0), (1,0))
        0.0
        >>> angle_between_2D_vectors((1, 1), (1,0))
        0.7853981633974483
        >>> angle_between_2D_vectors((-1, 1), (1,0))
        -0.7853981633974483
    """
    return math.atan2(*orig_2dvetor) - math.atan2(*dest_2dvector)

def find_yaw_to_align_XY_OBB_with_BB(scene):
    """
    Find the yaw rotation to parallel  X and Y planes of a Oriented Bounding Box with its Bounding Box
    :param scene: trimesh scene
    :return: yaw rotation to parallel X and Y planes
    """
    box_ori = scene.bounding_box_oriented.as_outline()
    obb_vertices = np.asarray(box_ori.vertices)

    base_num_corner = 5 # this was arbitrarily selected
    comb = [v for v in box_ori.vertex_nodes if base_num_corner in v]
    # initializing variable
    yaw = math.pi
    for c in comb:
        base_corner1 = obb_vertices[c[0]]
        base_corner2 = obb_vertices[c[1]]
        vect = base_corner1[:2] - base_corner2[:2]
        angle_y = angle_between_2D_vectors(vect, (1,0) )
        if np.linalg.norm(vect) > .5 and abs(yaw) > abs(angle_y):
            yaw = angle_y

    return yaw

def define_scene_boundary_on_the_fly(scene):
    rot_angle_1 = find_yaw_to_align_XY_OBB_with_BB(scene)

    scn = scene.copy(include_cache=True)
    scn.apply_transform(trimesh.transformations.euler_matrix(0,0,rot_angle_1,axes='rxyz'))
    bb_vertices = scn.bounding_box.vertices

    scene_max_x, scene_max_y = bb_vertices[:, :2].max(axis=0)
    scene_min_x, scene_min_y = bb_vertices[:, :2].min(axis=0)

    return rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y


def read_full_mesh_sdf(dataset_path, dataset, scene_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'prox':
        scene_mesh_path = os.path.join(dataset_path, 'scenes')
        scene = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_path, scene_name + '.ply'))
        cur_scene_verts = np.asarray(scene.vertices)

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)  # [1, 256, 256, 256]


    elif dataset == 'mp3d':
        scene = o3d.io.read_triangle_mesh(os.path.join(dataset_path, scene_name + '.ply'))
        # swap z, y axis
        cur_scene_verts = np.zeros(np.asarray(scene.vertices).shape)
        cur_scene_verts[:, 0] = np.asarray(scene.vertices)[:, 0]
        cur_scene_verts[:, 1] = np.asarray(scene.vertices)[:, 2]
        cur_scene_verts[:, 2] = np.asarray(scene.vertices)[:, 1]

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_min = np.array([grid_min[0], grid_min[2], grid_min[1]])
            grid_max = np.array([grid_max[0], grid_max[2], grid_max[1]])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = sdf.transpose(0, 2, 1)  # swap y,z axis
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)


    elif dataset == 'replica':
        scene = o3d.io.read_triangle_mesh(os.path.join(os.path.join(dataset_path, scene_name), 'mesh.ply'))
        cur_scene_verts = np.asarray(scene.vertices)

        ## read scene sdf
        scene_sdf_path = os.path.join(dataset_path, 'sdf')
        with open(os.path.join(scene_sdf_path, scene_name + '.json')) as f:
            sdf_data = json.load(f)
            grid_min = np.array(sdf_data['min'])
            grid_max = np.array(sdf_data['max'])
            grid_dim = sdf_data['dim']
        sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        s_grid_min_batch = torch.tensor(grid_min, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_grid_max_batch = torch.tensor(grid_max, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)
        s_sdf_batch = s_sdf_batch.repeat(1, 1, 1, 1)

    return scene, cur_scene_verts, s_grid_min_batch, s_grid_max_batch, s_sdf_batch




if __name__ == '__main__':
    import os
    dataset_prox_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data/datasets/prox/scenes"

    for scene in os.listdir(dataset_prox_dir):

        scn = trimesh.load(os.path.join(dataset_prox_dir, scene))

        rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y = define_scene_boundary_on_the_fly(scn)
        print(rot_angle_1, scene_min_x, scene_max_x, scene_min_y, scene_max_y)

        box = scn.bounding_box.as_outline()
        box_ori = scn.bounding_box_oriented.as_outline()
        obb_vertices = np.asarray(box_ori.vertices)


        base_num_corner = 5
        matched_num_corner = None
        comb = [v for v in box_ori.vertex_nodes if base_num_corner in v]
        min_angle = math.pi
        for c in comb:
            base_corner1 = obb_vertices[c[0]]
            base_corner2 = obb_vertices[c[1]]
            vect = base_corner1[:2] - base_corner2[:2]
            angle_y = angle_between_2D_vectors(vect)
            if np.linalg.norm(vect) > .5 and abs(min_angle) > abs(angle_y):
                matched_num_corner = c[1]
                min_angle = angle_y

        assert rot_angle_1 == min_angle

        # bounding box corner are always ordered
        base_corner1 = obb_vertices[base_num_corner]
        base_corner2 = obb_vertices[matched_num_corner]
        vect = base_corner1[:2] - base_corner2[:2]

        scn.apply_transform(trimesh.transformations.euler_matrix(0,0,min_angle,axes='rxyz'))
        aligned_box = scn.bounding_box.as_outline()



        s = trimesh.Scene()
        s.add_geometry(trimesh.primitives.Sphere(radius=.4, center=base_corner1 ))
        s.add_geometry(trimesh.primitives.Sphere(radius=.4, center=base_corner2 ))
        s.add_geometry(scn)
        s.add_geometry(box)
        # s.add_geometry(trimesh.load(os.path.join(dataset_prox_dir, scene)))
        s.add_geometry(box_ori)
        s.add_geometry(aligned_box)
        s.show(caption = scene)
