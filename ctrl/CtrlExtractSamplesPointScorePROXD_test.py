import json
import os
import warnings
from os.path import join as opj

import numpy as np
import pandas as pd
import trimesh
from prettytable import PrettyTable
from scipy.spatial import distance
from vedo import Lines, Spheres, load, colorMap, trimesh2vtk

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores
from util.util_mesh import read_sdf
from util.util_proxd import load_smplx_model, load_vposer_model, optimize_body_on_environment
from utils import get_contact_id
from view.ViewExtractSamplesPointScorePROXD_test import ViewExtractSamplesPointScorePROXD_test

warnings.simplefilter("ignore", UserWarning)

class CtrlExtractSamplesPointScorePROXD_test():
    visualized_samples = []
    scores_data = None
    view = None

    max_limit_score = None
    max_limit_missing = None
    max_limit_cv_collided = None

    np_points = None
    np_scores = None
    np_missings = None
    np_cv_vollided = None

    np_bad_normal_points = None

    np_full_points = None

    affordance_name = None
    tester = None

    smplx_model = None
    vposer_model = None
    body_gender =  None

    def __init__(self, trainings_dir, json_conf_execution_file,
                 env_test_results_dir, prop_configs_dir, smplx_model_dir, vposer_model_dir,
                 datasets_dir, dataset_name, env_test_name):


        self.tester = TesterClearance(trainings_dir, json_conf_execution_file)
        self.affordance_name = self.tester.affordances[0][0]
        affordance_object = self.tester.affordances[0][1]
        subdir_name = self.affordance_name + "_" + affordance_object
        env_test_results = opj(env_test_results_dir, self.affordance_name)


        for dataset in os.listdir(datasets_dir):
            for env_file in os.listdir(opj(datasets_dir, dataset, "scenes")):
                if env_file.startswith(env_test_name):
                    self.file_mesh_env = opj(datasets_dir, dataset,"scenes", env_file)
                    break

        propagation_settings_file = opj(prop_configs_dir, subdir_name, 'propagation_data.json')
        with open(propagation_settings_file) as json_file:
            propagation_settings = json.load(json_file)
        self.max_limit_score = propagation_settings['max_limit_score']
        self.max_limit_missing = propagation_settings['max_limit_missing']
        self.max_limit_cv_collided = propagation_settings['max_limit_cv_collided']

        df_scores_data = pd.read_csv(opj(env_test_results, "test_scores.csv"))
        self.scores_data = FullDataClearanceScores(df_scores_data, self.affordance_name)

        self.view = ViewExtractSamplesPointScorePROXD_test(self, self.file_mesh_env)

        self.np_points, self.np_scores, self.np_missings, self.np_cv_collided = self.scores_data.filter_data_scores(
            self.max_limit_score,
            self.max_limit_missing,
            self.max_limit_cv_collided)
        self.view.add_point_cloud(self.np_points, self.np_scores, at=0)

        # draw sampled points with environment BAD normal
        self.np_bad_normal_points = self.scores_data.np_bad_normal_points
        scores = np.zeros(self.np_bad_normal_points.shape[0])
        scores.fill(self.max_limit_score)
        self.view.add_point_cloud(self.np_bad_normal_points, scores, at=0)

        self.np_full_points = np.concatenate((self.np_points, self.np_bad_normal_points), axis=0)
        self.np_full_scores = np.concatenate((self.np_scores, scores), axis=0)

        self.body_gender = self.tester.it_descriptor.definition["extra"]["body_gender"]
        self.contact_regions = self.tester.it_descriptor.definition["extra"]["contact_regions"]

        self.smplx_model = load_smplx_model(smplx_model_dir, self.body_gender)
        self.vposer_model = load_vposer_model(vposer_model_dir)

        body_params_file = self.tester.it_descriptor.object_filename()[:-11]+"_smplx_body_params.npy"
        self.np_body_params = np.load(body_params_file)

        self.datasets_dir = datasets_dir
        self.dataset_dir = opj(datasets_dir, dataset_name)
        self.prox_dataset_dir = opj(datasets_dir, "prox")
        self.env_test_name = env_test_name

        self.s_grid_min_batch, self.s_grid_max_batch, self.s_sdf_batch = read_sdf(self.dataset_dir, self.env_test_name)


    def start_viewer(self, saving_output=False):
        self.view.start(saving_output)

    def get_data_from_nearest_point_to(self, np_point):
        closest_index = distance.cdist([np_point], self.np_full_points).argmin()
        np_nearest_point = self.np_full_points[closest_index]
        np_nearest_score = self.np_full_scores[closest_index]

        best_angle=None

        self.view.add_point_cloud([np_nearest_point], [np_nearest_score], 20, at=0)

        df_point_data = self.scores_data.get_point_data(np_nearest_point)

        print("\nInteraction: ", self.affordance_name)
        str_point = "({:.4f}, {:.4f}, {:.4f})".format(np_nearest_point[0], np_nearest_point[1], np_nearest_point[2])
        print("Point: ", str_point)
        if closest_index >= self.np_points.shape[0]:
            table = PrettyTable()
            table.field_names = ["default_score", "info"]
            info = 'bad environment normal orientation'
            table.add_row([self.max_limit_score, info])
            print(table)

        else:
            table = PrettyTable()
            table.field_names = ["orientation", "cv collisions", "missing", "score", "info"]
            for row_index, row in df_point_data.iterrows():
                # precalculated data
                ori = int(row['orientation'])
                angle = row['angle']
                score = row['score']
                cv_collisions = int(row['cv_collided'])
                missings = int(row['missings'])

                if (self.np_scores[closest_index] == score
                        and self.np_missings[closest_index] == missings
                        and self.np_cv_collided[closest_index] == cv_collisions):
                    info = 'SELECTED'
                    best_angle = angle
                else:
                    info = ''
                table.add_row([ori, cv_collisions, missings, score, info])

                # provenance vectors
                idx_from = ori * self.tester.num_pv
                idx_to = idx_from + self.tester.num_pv
                pv_points = self.tester.compiled_pv_begin[idx_from:idx_to]
                pv_vectors = self.tester.compiled_pv_direction[idx_from:idx_to]
                # clearance vectors
                idx_from = ori * self.tester.num_cv
                idx_to = idx_from + self.tester.num_cv
                cv_points = self.tester.compiled_cv_begin[idx_from:idx_to]
                cv_vectors = self.tester.compiled_cv_direction[idx_from:idx_to]

                provenance_vectors = Lines(np_nearest_point+pv_points, np_nearest_point+pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
                clearance_vectors = Lines(np_nearest_point+cv_points, np_nearest_point+cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
                cv_from = Spheres(np_nearest_point+cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

                vedo_obj = load(self.tester.objs_filenames[0]).lighting("plastic")
                vedo_obj.c(colorMap(score, name='jet', vmin=0, vmax=self.max_limit_score))
                if cv_collisions > self.max_limit_cv_collided or missings > self.max_limit_missing:
                    if cv_collisions > self.max_limit_cv_collided:
                        vedo_obj.alpha(.05)
                        provenance_vectors.alpha(.05)
                        clearance_vectors.alpha(.05)
                        cv_from.alpha(.05)
                    if missings > self.max_limit_missing:
                        vedo_obj.c('black')
                        vedo_obj.alpha(.25)
                        provenance_vectors.alpha(.25)
                        clearance_vectors.alpha(.25)
                        cv_from.alpha(.25)
                else:
                    a = max([1 - cv_collisions * (1 / (self.max_limit_cv_collided + 1)), .20])
                    vedo_obj.alpha(a)
                    provenance_vectors.alpha(a)
                    clearance_vectors.alpha(a)
                    cv_from.alpha(a)

                vedo_obj.rotateZ(angle, rad=True)
                vedo_obj.pos(x=np_nearest_point[0], y=np_nearest_point[1], z=np_nearest_point[2])

                if self.view.started:
                    self.view.add_vedo_element(provenance_vectors, at=0)
                    self.view.add_vedo_element(clearance_vectors, at=0)
                    self.view.add_vedo_element(cv_from, at=0)
                    self.view.add_vedo_element(vedo_obj, at=0)
            print(table)

        return np_nearest_point, best_angle

    def optimize_best_scored_position(self, np_point, best_angle, view_evolution_screens=False):

        if self.view.started:
            vtk_object = load(self.tester.it_descriptor.object_filename())
            vtk_object.c([150, 150, 0])
            vtk_object.rotate(best_angle, axis=(0, 0, 1), rad=True)
            vtk_object.pos(x=np_point[0], y=np_point[1], z=np_point[2])

            self.view.add_vedo_element(vtk_object, at=1)

        scene_trimesh = trimesh.load(self.file_mesh_env)
        contact_ids, _ = get_contact_id(body_segments_folder=opj(self.prox_dataset_dir, 'body_segments'),
                                        contact_body_parts=self.contact_regions)

        body_trimesh_optim, np_body_params_optim = optimize_body_on_environment(
                                        scene_trimesh, self.s_grid_min_batch, self.s_grid_max_batch, self.s_sdf_batch,
                                        self.smplx_model, self.vposer_model,
                                        self.np_body_params, np_point, best_angle, contact_ids,
                                        weight_loss_rec_verts=1.0,
                                        weight_loss_rec_bps=1.0,
                                        weight_loss_vposer=0.02,
                                        weight_loss_shape=0.01,
                                        weight_loss_hand=0.01,
                                        weight_collision=8.0,
                                        weight_loss_contact=0.5,
                                        itr_s2=150,
                                        view_evolution_screens=view_evolution_screens)

        if self.view.started:
            body_vedo_optim = trimesh2vtk(body_trimesh_optim)
            self.view.add_vedo_element(body_vedo_optim, at=1)

        return body_trimesh_optim, np_body_params_optim

