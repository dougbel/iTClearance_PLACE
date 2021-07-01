import json
import os
import time

import numpy as np
import torch
import vedo
from mesh_to_sdf import mesh_to_voxels, scale_to_unit_cube
import trimesh


if __name__ == "__main__":

    visualize = True
    save_results = False
    grid_dim = 10

    # dataset_name = "prox"
    # dataset_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}"
    # scenes_dir = f"{dataset_dir}/scenes"
    # sdf_dir = f"{dataset_dir}/sdf_tmp"


    dataset_name = "replica_v1"
    dataset_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}"
    scenes_dir = f"{dataset_dir}/scenes_downsampled"
    sdf_dir = f"{dataset_dir}/sdf_tmp"

    for scene_file in os.listdir(scenes_dir): # ['MPH1Library.ply']:

        print(scene_file)

        scene_name = scene_file[:scene_file.find(".ply")]

        mesh_orig = trimesh.load(os.path.join(scenes_dir, scene_file))
        # vedo_orig = vedo.vtk2trimesh(vedo_orig.decimate(N=50000, method='pro', boundaries=True))
        vedo_orig = vedo.load(os.path.join(scenes_dir, scene_file))

        transform = np.eye(4)
        # translate to center of axis aligned bounds
        transform[:3, 3] = mesh_orig.bounds.mean(axis=0)
        bb_padded = trimesh.primitives.Box(transform=transform, extents=mesh_orig.extents*1.5,mutable=False)

        np_min = np.array(bb_padded.vertices.min(axis=0))
        np_max = np.array(bb_padded.vertices.max(axis=0))

        step_x, step_y, step_z = (np_max - np_min) / grid_dim
        init_x = step_x/2 + np_min[0]
        init_y = step_y/2 + np_min[1]
        init_z = step_z/2 + np_min[2]


        points = np.empty((grid_dim,grid_dim,grid_dim,3))

        for xi in range(grid_dim):
            for yi in range(grid_dim):
                for zi in range(grid_dim):
                    points[xi,yi,zi] = np.array( [init_x+xi*step_x, init_y+yi*step_y, init_z+zi*step_z])


        start = time.time()
        voxel_points = points.reshape(grid_dim * grid_dim * grid_dim, 3)
        sdf_values = np.empty(voxel_points.shape[0])
        (closest_points_in_env, norms, id_triangle) = mesh_orig.nearest.on_surface(voxel_points)

        if visualize:
            v = vedo.Plotter()
            vedo_orig.backFaceCulling(True)
            v.add(vedo_orig)
            v.add(vedo.trimesh2vtk(bb_padded.as_outline()))
            v.add(vedo.Points(voxel_points, c="green"))


        for i in range(len(id_triangle)):
            v_p = voxel_points[i]
            s_p = closest_points_in_env[i]
            s_n = mesh_orig.face_normals[id_triangle[i]]
            sign = -1 if np.dot( (v_p - s_p ), s_n ) < 0 else 1
            sdf_values[i] =  sign * norms[i]

            if visualize:
                c= "red" if sign<0 else "green"
                v.add(vedo.Line(s_p, v_p, c=c))
                v.add(vedo.Line(s_p, s_p+s_n*.1, c="black"))

        if visualize:
            v.show(axes=1, interactive=True)


        print(time.time() - start, " seconds for ", len(points) ," point")
        v.close()


        if save_results:
            if not os.path.exists(sdf_dir):
                os.makedirs(sdf_dir)

            np.save(os.path.join(sdf_dir, scene_name+"_sdf.npy"), sdf_values)
            dictionary = {
                "max": list(np_max),
                "dim": grid_dim,
                "min": list(np_min)
            }

            with open(os.path.join(sdf_dir, scene_name+".json"), "w") as outfile:
                json.dump(dictionary, outfile)

        # sdf_data = np.load(os.path.join(sdf_dir, scene_name+"_sdf.npy")).reshape(grid_dim, grid_dim, grid_dim)
        # # between x, z
        # sdf_data = sdf_data.swapaxes(0, 2)
        #
        # from vedo import Volume, show, volumeFromMesh
        # vol = Volume(sdf_data)
        # vol.addScalarBar3D()
        # iso = vol.isosurface(0, True)
        # # iso.flipNormals()
        # iso.backFaceCulling(False)
        # X, Y, Z = np.mgrid[:30, :30, :30]
        # show([iso, vedo_orig], axes=1).close()
