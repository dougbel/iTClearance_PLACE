"""I moved some milimeters up some training cause the gap between the bodies and teh environment was so small for a
cosidered GOD TRAINING, then a training process is performed one more time """


import json
import os
from os.path import join as opj

import numpy as np
from vedo import Lines, Spheres, Text2D
from vedo.utils import flatten

from it import util
from it.training.ibs import IBSMesh
from it.training.maxdistancescalculator import MaxDistancesCalculator
from it.training.sampler import OnGivenPointCloudWeightedSampler
from it_clearance.testing.deglomerator import DeglomeratorClearance
from it_clearance.training.agglomerator import AgglomeratorClearance
from it_clearance.training.sampler import PropagateNormalObjectPoissonDiscSamplerClearance
from it_clearance.training.saver import SaverClearance
from it_clearance.training.trainer import TrainerClearance
from it_clearance.utils import get_vtk_items_cv_pv
from util.util_interactive import SelectorITClearanceReferencePoint
from util.util_mesh import find_yaw_to_align_XY_OBB_with_BB
from util.util_proxd import get_vertices_from_body_params, translate_smplx_body, load_smplx_model, rotate_smplx_body, \
    load_vposer_model

if __name__ == "__main__":

    interaction= "sitting_looking_to_right"

    datasets_dir = "/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/data"

    smplx_model_path = opj(datasets_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_path = opj(datasets_dir, "pretrained_place", "body_models", "vposer_v1_0")

    descriptors_dir = "output/descriptors_repository_v2"

    env_file = None
    obj_file = None
    json_training_file = None
    np_body_params_file = None

    interactions = os.listdir(descriptors_dir)
    interactions.sort()

    for descriptor in interactions:
        if interaction is not None and descriptor != interaction:
            continue

        sub_dir = opj(descriptors_dir, descriptor)

        obj_file_name = [f for f in os.listdir(opj(descriptors_dir, descriptor)) if f.endswith("_object.ply")][0]

        prefix_file_name = obj_file_name[:obj_file_name.find("_object.ply")]

        obj_file = opj(sub_dir, prefix_file_name + "_object.ply")
        env_file = opj(sub_dir, prefix_file_name + "_environment.ply")
        json_training_file = opj(sub_dir, prefix_file_name + ".json")
        np_body_params_file = opj(sub_dir, prefix_file_name + "_smplx_body_params.npy")

        print("Fixing ", descriptor)
        print("environment ", env_file)
        print("object ", obj_file)
        print("json_training_file ", json_training_file)
        print("np_body_params_file ", np_body_params_file)

        import vedo
        import trimesh
        trimesh_env = trimesh.load(env_file)
        trimesh_env.visual.face_colors=[100,100,100,255]
        vedo_env = vedo.trimesh2vtk(trimesh_env)
        vedo_env.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))

        trimesh_obj = trimesh.load(obj_file)
        trimesh_obj.visual.face_colors = [0, 100, 0, 100]
        vedo_obj = vedo.trimesh2vtk(trimesh_obj)
        # vedo_obj.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))

        # trimesh_ibs = trimesh.load(ibs_file)
        # trimesh_ibs.visual.face_colors = [0, 0, 100, 100]
        # vedo_ibs = vedo.trimesh2vtk(trimesh_ibs)

        with open(json_training_file) as f:
            train_data = json.load(f)
        it_descriptor = DeglomeratorClearance(sub_dir, train_data['affordance_name'], train_data['obj_name'])
        num_cv = train_data['trainer']['cv_sampler']['sample_clearance_size']
        cv_points = it_descriptor.cv_points[0:num_cv]
        cv_vectors = it_descriptor.cv_vectors[0:num_cv]
        clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
        cv_from = Spheres(cv_points, r=.004, c="yellow", alpha=1).lighting("plastic")
        num_pv = train_data['trainer']['sampler']['sample_size']
        pv_points = it_descriptor.pv_points[0:num_pv]
        pv_vectors = it_descriptor.pv_vectors[0:num_pv]
        provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")

        selected_p=None
        while selected_p is None:
            sel_gui = SelectorITClearanceReferencePoint(vedo_env, vedo_obj, provenance_vectors)
            selected_p = sel_gui.select_reference_point_to_train()


        print("Translating env and obj meshes")
        trimesh_env.apply_translation(-selected_p)
        trimesh_obj.apply_translation(-selected_p)

        rot_angle_1 = find_yaw_to_align_XY_OBB_with_BB(trimesh_env)
        trimesh_env.apply_transform(trimesh.transformations.euler_matrix(0, 0, rot_angle_1, axes='rxyz'))
        trimesh_obj.apply_transform(trimesh.transformations.euler_matrix(0, 0, rot_angle_1, axes='rxyz'))

        np_body_params = np.load(np_body_params_file)
        smplx_model = load_smplx_model(smplx_model_path, train_data["extra"]["body_gender"])
        np_body_params = translate_smplx_body(np_body_params, smplx_model, -selected_p)
        np_body_params = rotate_smplx_body(np_body_params, smplx_model, rot_angle_1)

        train_data["extra"]["transform_for_training"] = {"reference_point": list(selected_p),
                                                       "XY_alignment_Z_rotation": rot_angle_1}


        affordance_name = train_data['affordance_name']
        env_name = train_data['env_name']
        obj_name = train_data['obj_name']
        influence_radio_bb = 2
        extension, middle_point = util.influence_sphere(trimesh_obj, influence_radio_bb)
        tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(trimesh_env, middle_point, extension)

        # vp = vedo.Plotter(bg="white", size=(800, 600), axes=9)
        # vp.add(vedo.trimesh2vtk(trimesh_env))
        # vp.add(vedo.trimesh2vtk(trimesh_obj))
        # vp.show()
        # vp.close()

        print("Calculating IBS")
        ibs_init_size_sampling = train_data["ibs_calculator"]["init_size_sampling"]  # 400
        ibs_resamplings = train_data["ibs_calculator"]["resamplings"]  # 4
        sampler_rate_ibs_samples = 5
        sampler_rate_generated_random_numbers = train_data["trainer"]["sampler"]["rate_generated_random_numbers"]
        influence_radio_ratio = 1.2
        ibs_calculator = IBSMesh(ibs_init_size_sampling, ibs_resamplings)
        ibs_calculator.execute(tri_mesh_env_segmented, trimesh_obj)
        tri_mesh_ibs = ibs_calculator.get_trimesh()
        sphere_ro, sphere_center = util.influence_sphere(trimesh_obj, influence_radio_ratio)
        tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)
        np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]


        print("Training iT with Clearance Vectors")
        pv_sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                                      rate_generated_random_numbers=sampler_rate_generated_random_numbers)
        cv_sampler = PropagateNormalObjectPoissonDiscSamplerClearance()
        trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=trimesh_env,
                                   tri_mesh_obj=trimesh_obj, pv_sampler=pv_sampler, cv_sampler=cv_sampler)
        agglomerator = AgglomeratorClearance(trainer, num_orientations=8)

        max_distances = MaxDistancesCalculator(pv_points=trainer.pv_points, pv_vectors=trainer.pv_vectors,
                                               tri_mesh_obj=trimesh_obj, consider_collision_with_object=True,
                                               radio_ratio=influence_radio_ratio)

        output_subdir = "IBSMesh_" + str(ibs_init_size_sampling) + "_" + str(ibs_resamplings) + "_"
        output_subdir += pv_sampler.__class__.__name__ + "_" + str(sampler_rate_ibs_samples) + "_"
        output_subdir += str(sampler_rate_generated_random_numbers) + "_"
        output_subdir += cv_sampler.__class__.__name__ + "_" + str(cv_sampler.sample_size)


        # #########    SAVING    ###############

        print( "Saving")
        saver = SaverClearance(affordance_name, env_name, obj_name, agglomerator,
                       max_distances, ibs_calculator, trimesh_obj, output_subdir)

        with open(opj(saver.output_dir, affordance_name, prefix_file_name + ".json"), 'w') as fp:
            json.dump(train_data, fp, indent=4, sort_keys=True)

        np.save(opj(saver.output_dir, affordance_name, prefix_file_name + "_smplx_body_params.npy"), np_body_params)


        # #########    VISUALISATION    ###############

        vedo_items = get_vtk_items_cv_pv(trainer.pv_points, trainer.pv_vectors, trainer.cv_points, trainer.cv_vectors,
                                         trimesh_obj=trimesh_obj,
                                         trimesh_ibs=tri_mesh_ibs_segmented)

        vposer_model = load_vposer_model(vposer_model_path)
        np_body_verts_sample = get_vertices_from_body_params(smplx_model, vposer_model, np_body_params)
        body_trimesh_proxd = trimesh.Trimesh(np_body_verts_sample, faces=smplx_model.faces)
        body_trimesh_proxd.visual.face_colors = [255, 255, 255, 255]
        body_vedo_proxd = vedo.trimesh2vtk(body_trimesh_proxd)


        vp = vedo.Plotter(bg="white", size=(800, 600), axes=2)
        vedo_env = vedo.trimesh2vtk(trimesh_env)
        vedo_txt = Text2D(affordance_name, pos="top-left",
                          bg='darkblue', c="lightgray", font='Arial', s=0.8, alpha=0.9)
        vedo_env.lighting(ambient=0.8, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))
        vp.show(flatten([vedo_items, body_vedo_proxd, vedo_env, vedo_txt]))
        vp.close()