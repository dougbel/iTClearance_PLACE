import math

import numpy as np

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


if __name__ == '__main__':
    import trimesh
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