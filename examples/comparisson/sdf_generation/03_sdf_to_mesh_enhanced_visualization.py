
import json
import os
import time

import numpy as np
import torch
import vedo
from mesh_to_sdf import mesh_to_voxels, scale_to_unit_cube
import trimesh


if __name__ == "__main__":

    grid_dim =256

    dataset_name = "replica_v1"
    dataset_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}"
    scenes_dir = f"{dataset_dir}/scenes"
    sdf_dir = f"{dataset_dir}/sdf"

    iso_distance = 0.1

    # dataset_name = "replica_v1"
    # dataset_dir = f"/media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/{dataset_name}"
    # scenes_dir = f"{dataset_dir}/scenes_downsampled"
    # sdf_dir = f"{dataset_dir}/sdf_tmp"

    for scene_file in os.listdir(scenes_dir):  # ['MPH1Library.ply']:

        print(scene_file)
        scene_name = scene_file[:scene_file.find(".ply")]
        vedo_orig = vedo.load(os.path.join(scenes_dir, scene_file))
        sdf_data = np.load(os.path.join(sdf_dir, scene_name+"_sdf.npy")).reshape(grid_dim, grid_dim, grid_dim)

        # between x, z
        sdf_data = sdf_data.swapaxes(0, 2)

        from vedo import Volume, show, volumeFromMesh
        vol = Volume(sdf_data)
        vol.addScalarBar3D()
        iso = vol.isosurface(0, True).reverse(True, True).c([0,255,0]).backColor([0,255,0]).backFaceCulling(True)
        iso_cc = vol.isosurface(0, True).reverse(True, True).c([0, 255, 0]).backColor([0, 255, 0])
        iso_plus = vol.isosurface(iso_distance, True).c([0,0,255]).alpha(.5)
        iso_minus = vol.isosurface(-iso_distance, True).c([255, 0, 0]).alpha(.5)



        vp1 = vedo.Plotter(axes=1)
        vedo_orig.backFaceCulling(True)
        vp1.show(vedo_orig, interactive=False)

        vp2 = vedo.Plotter(shape=(1,3), axes=1)

        vp2.show(iso, at=0)
        vp2.show([iso_cc, iso_plus], f"positive iso ({iso_distance})", at=1)
        vp2.show([iso_cc, iso_minus], f"negative iso ({-iso_distance})", at=2)
        vedo.interactive()
        vp1.close()
        vp2.close()