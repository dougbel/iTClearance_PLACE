import json
import os
from collections import Counter

import pandas as pd
import numpy as np
from  os.path import join as opj

from vedo import load, Points, Plotter

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores
from view.sampler import ViewSampler


class CtrlPropagatorSampler:

    def __init__(self, directory_of_trainings, json_conf_execution_file,
                 directory_env_test_results, directory_of_prop_configs, file_mesh_env):

        self.tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
        self.affordance_name = self.tester.affordances[0][0]
        affordance_object = self.tester.affordances[0][1]
        subdir_name = self.affordance_name + "_" + affordance_object
        env_test_results = opj(directory_env_test_results, self.affordance_name)
        self.file_mesh_env = file_mesh_env
        self.json_file_propagation = opj(directory_of_prop_configs, subdir_name, 'propagation_data.json')
        self.ply_obj_file = opj(directory_of_trainings, self.affordance_name, subdir_name + "_object.ply")
        self.csv_file_scores = opj(directory_env_test_results,self.affordance_name,  "test_scores.csv")

        self.np_pc_tested, self.np_scores = self.map_scores()


        self.votes = None
        self.pd_best_scores = None
        self.idx_votes = -1
        self.vtk_samples = []

    def map_scores(self):

        with open(self.json_file_propagation) as json_file:
            propagation_settings = json.load(json_file)

        max_limit_score = propagation_settings['max_limit_score']
        max_limit_missing = propagation_settings['max_limit_missing']
        max_limit_cv_collided = propagation_settings['max_limit_cv_collided']
        epsilon = propagation_settings['epsilon']
        function = propagation_settings['function']

        results_it_test = pd.read_csv(self.csv_file_scores)

        scores_data = FullDataClearanceScores(results_it_test, self.affordance_name)

        bad_normal_points = scores_data.np_bad_normal_points
        bad_normal_scores = np.zeros(scores_data.np_bad_normal_points.shape[0])

        np_filtered_ps, np_filtered_scores, __, __ = scores_data.filter_data_scores(max_limit_score,
                                                                                    max_limit_missing,
                                                                                    max_limit_cv_collided)
        # MAPPING SCORES TO [0,1]
        np_filtered_scores_mapped = [-value_in / max_limit_score + 1 for value_in in np_filtered_scores]

        np_full_points = np.concatenate((np_filtered_ps, bad_normal_points), axis=0)
        np_full_scores = np.concatenate((np_filtered_scores_mapped, bad_normal_scores), axis=0)

        return np_full_points, np_full_scores

    def filter_data_scores(self):
        with open(self.json_file_propagation) as f:
            propagation_data = json.load(f)

        max_score = propagation_data['max_limit_score']
        max_missing = propagation_data['max_limit_missing']
        max_collided = propagation_data['max_limit_cv_collided']

        pd_scores = pd.read_csv(self.csv_file_scores)

        filtered_df = pd_scores.loc[(pd_scores.score.notnull()) &  # avoiding null scores (bar normal environment)
                                    (pd_scores.missings <= max_missing) &  # avoiding scores with more than max missing
                                    (pd_scores.score <= max_score) &
                                    (pd_scores.cv_collided <= max_collided),
                                    pd_scores.columns != 'interaction']  # returning all columns but interaction name

        return filtered_df.loc[filtered_df.groupby(['point_x', 'point_y', 'point_z'])['score'].idxmin()]

    def get_sample(self, visualize = True):
        if self.votes is None:
            self.pd_best_scores = self.filter_data_scores()
            self.votes = self.generate_votes()

        while True:
            self.idx_votes += 1
            idx_sample = self.votes[self.idx_votes][0]
            point_sample = self.np_pc_tested[idx_sample]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            vtk_object = None
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.c([200,200,200])
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                print(point_sample, " at ", angle_sample)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                self.vtk_samples.append(vtk_object)
                break

        if visualize:
            vp = Plotter(verbose=0, title="Scores", bg="white", size=(1200, 800))
            vedo_file_env = load(self.file_mesh_env)
            vp.add(vedo_file_env)
            pts = Points(self.np_pc_tested, r=5)
            pts.cellColors(self.np_scores, cmap='jet_r', vmin=0, vmax=1)
            pts.addScalarBar(c='jet_r', nlabels=5, pos=(0.8, 0.25))
            vp.add(pts)
            vp.add(vtk_object)
            vp.show()

        return vtk_object, point_sample

    def generate_votes(self):
        sum_mapped_norms = sum(self.np_scores)
        probabilities = [float(score) / sum_mapped_norms for score in self.np_scores]
        n_rolls = 10 * self.np_scores.shape[0]
        rolls = np.random.choice(self.np_scores.shape[0], n_rolls, p=probabilities)
        return Counter(rolls).most_common()

    def angle_with_best_score(self, x, y, z):
        angles = self.pd_best_scores[(self.pd_best_scores['point_x'].round(decimals=5) == round(x, 5)) &
                                     (self.pd_best_scores['point_y'].round(decimals=5) == round(y, 5)) &
                                     (self.pd_best_scores['point_z'].round(decimals=5) == round(z, 5))].angle

        return angles.array[0] if (angles.shape[0] == 1) else -1