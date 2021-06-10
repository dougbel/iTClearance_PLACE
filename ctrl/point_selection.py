import json
import os
import gc

import pandas as pd
import numpy as np
from prettytable import PrettyTable

from scipy.spatial import distance
from vedo import Lines, Spheres, load, colorMap, Text2D

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores
from view.point_selection import ViewPointSelection


class ControlPointSelection:
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

    def __init__(self, directory_of_trainings, json_conf_execution_file,
                 directory_env_test_results, directory_of_prop_configs, file_mesh_env):
        self.tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
        self.affordance_name = self.tester.affordances[0][0]
        affordance_object = self.tester.affordances[0][1]
        subdir_name = self.affordance_name + "_" + affordance_object
        env_test_results = os.path.join(directory_env_test_results, self.affordance_name)
        self.file_mesh_env = file_mesh_env

        propagation_settings_file = os.path.join(directory_of_prop_configs, subdir_name, 'propagation_data.json')
        with open(propagation_settings_file) as json_file:
            propagation_settings = json.load(json_file)
        self.max_limit_score = propagation_settings['max_limit_score']
        self.max_limit_missing = propagation_settings['max_limit_missing']
        self.max_limit_cv_collided = propagation_settings['max_limit_cv_collided']

        df_scores_data = pd.read_csv(os.path.join(env_test_results, "test_scores.csv"))
        self.scores_data = FullDataClearanceScores(df_scores_data, self.affordance_name)

        self.view = ViewPointSelection(self, self.file_mesh_env)

        self.view_point_cloud_test_results()

        self.points_selected = []
        self.vedo_best_bodies = []

    def view_point_cloud_test_results(self):
        self.np_points, self.np_scores, self.np_missings, self.np_cv_collided = self.scores_data.filter_data_scores(
            self.max_limit_score,
            self.max_limit_missing,
            self.max_limit_cv_collided)
        self.view.add_point_cloud(self.np_points, self.np_scores)

        # draw sampled points with environment BAD normal
        self.np_bad_normal_points = self.scores_data.np_bad_normal_points
        scores = np.zeros(self.np_bad_normal_points.shape[0])
        scores.fill(self.max_limit_score)
        self.view.add_point_cloud(self.np_bad_normal_points, scores)

        self.np_full_points = np.concatenate((self.np_points, self.np_bad_normal_points), axis=0)
        self.np_full_scores = np.concatenate((self.np_scores, scores), axis=0)

        txt = Text2D(self.affordance_name, pos="top-left",
                     bg='darkblue', c="lightgray", font='Arial', s=0.8, alpha=0.9)

        self.view.vp.add(txt)

    def start(self):
        self.view.show()

    def remove_body(self, mesh):
        print(mesh.picked3d)
        for i in range(len(self.points_selected)):
            if self.vedo_best_bodies[i] == mesh:
                self.points_selected.pop(i)
                self.vedo_best_bodies.pop(i)
                self.view.vp.clear()

                to_recover = [point for point in self.points_selected]
                self.points_selected.clear()
                self.vedo_best_bodies.clear()

                gc.collect()

                self.view.vp.add(self.view.vedo_file_env)
                self.view_point_cloud_test_results()
                for j in range(len(to_recover)):
                    self.get_data_from_nearest_point_to(to_recover[j])

                break

    def get_data_from_nearest_point_to(self, np_point):
        closest_index = distance.cdist([np_point], self.np_full_points).argmin()
        np_nearest_point = self.np_full_points[closest_index]
        np_nearest_score = self.np_full_scores[closest_index]

        self.view.add_point_cloud([np_nearest_point], [np_nearest_score], 20)

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
                else:
                    info = ''
                table.add_row([ori, cv_collisions, missings, score, info])

                if info != 'SELECTED':
                    continue

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
                # if cv_collisions > self.max_limit_cv_collided or missings > self.max_limit_missing:
                #     if cv_collisions > self.max_limit_cv_collided:
                #         vedo_obj.alpha(.05)
                #         provenance_vectors.alpha(.05)
                #         clearance_vectors.alpha(.05)
                #         cv_from.alpha(.05)
                #     if missings > self.max_limit_missing:
                #         vedo_obj.c('black')
                #         vedo_obj.alpha(.25)
                #         provenance_vectors.alpha(.25)
                #         clearance_vectors.alpha(.25)
                #         cv_from.alpha(.25)
                # else:
                #     a = max([1 - cv_collisions * (1 / (self.max_limit_cv_collided + 1)), .20])
                #     vedo_obj.alpha(a)
                #     provenance_vectors.alpha(a)
                #     clearance_vectors.alpha(a)
                #     cv_from.alpha(a)

                vedo_obj.rotateZ(angle, rad=True)
                vedo_obj.pos(x=np_nearest_point[0], y=np_nearest_point[1], z=np_nearest_point[2])

                self.view.vp.add(provenance_vectors)
                self.view.vp.add(clearance_vectors)
                self.view.vp.add(cv_from)
                self.view.vp.add(vedo_obj)

                self.points_selected.append(np_nearest_point)
                self.vedo_best_bodies.append(vedo_obj)

            print(table)