import json
import os
from collections import Counter

import pandas as pd
import random
import numpy as np
import vedo.shapes
from sklearn.cluster import KMeans
from  os.path import join as opj

from distinctipy import distinctipy
from vedo import load, Points, Plotter, Sphere

from it_clearance.testing.tester import TesterClearance
from si.fulldataclearancescores import FullDataClearanceScores



class CtrlPropagatorSampler:

    def __init__(self, directory_of_trainings, json_conf_execution_file,
                 directory_env_test_results, directory_of_prop_configs, file_mesh_env):

        self.tester = TesterClearance(directory_of_trainings, json_conf_execution_file)
        self.affordance_name = self.tester.affordances[0][0]
        affordance_object = self.tester.affordances[0][1]
        subdir_name = self.affordance_name + "_" + affordance_object
        self.file_mesh_env = file_mesh_env
        self.json_file_propagation = opj(directory_of_prop_configs, subdir_name, 'propagation_data.json')
        self.ply_obj_file = opj(directory_of_trainings, self.affordance_name, subdir_name + "_object.ply")
        self.csv_file_scores = opj(directory_env_test_results,self.affordance_name,  "test_scores.csv")

        self.np_pc_tested, self.np_mapped_scores = self.map_scores()
        self.pd_best_scores = self.get_best_score_by_tested_point()
        self.votes = self.generate_votes(self.np_mapped_scores)
        self.idx_votes = -1

    def map_scores(self):

        with open(self.json_file_propagation) as json_file:
            propagation_settings = json.load(json_file)

        max_limit_score = propagation_settings['max_limit_score']
        max_limit_missing = propagation_settings['max_limit_missing']
        max_limit_cv_collided = propagation_settings['max_limit_cv_collided']

        results_it_test = pd.read_csv(self.csv_file_scores)

        scores_data = FullDataClearanceScores(results_it_test, self.affordance_name)

        # bad_normal_points = scores_data.np_bad_normal_points
        # bad_normal_scores = np.zeros(scores_data.np_bad_normal_points.shape[0])

        np_filtered_ps, np_filtered_scores, __, __ = scores_data.filter_data_scores(max_limit_score,
                                                                                    max_limit_missing,
                                                                                    max_limit_cv_collided)
        # MAPPING SCORES TO [0,1]
        np_filtered_scores_mapped = np.asarray([-value_in / max_limit_score + 1 for value_in in np_filtered_scores])

        # np_full_points = np.concatenate((np_filtered_ps, bad_normal_points), axis=0)
        # np_full_mapped_scores = np.concatenate((np_filtered_scores_mapped, bad_normal_scores), axis=0)

        return np_filtered_ps, np_filtered_scores_mapped

    def get_best_score_by_tested_point(self):
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

    def generate_votes(self, np_mapped_scores):
        sum_mapped_norms = sum(np_mapped_scores)
        if sum_mapped_norms > 0:
            probabilities = [float(score) / sum_mapped_norms for score in np_mapped_scores]
            n_rolls = 10 * np_mapped_scores.shape[0]
            rolls = np.random.choice(np_mapped_scores.shape[0], n_rolls, p=probabilities)
            return Counter(rolls).most_common()
        else:
            return None

    def angle_with_best_score(self, x, y, z):
        angles = self.pd_best_scores[(self.pd_best_scores['point_x'].round(decimals=5) == round(x, 5)) &
                                     (self.pd_best_scores['point_y'].round(decimals=5) == round(y, 5)) &
                                     (self.pd_best_scores['point_z'].round(decimals=5) == round(z, 5))].angle

        return angles.array[0] if (angles.shape[0] == 1) else -1


    def get_sample(self, visualize = True):
        """
        Get a sample by using a votes system, the highest similarity the best possibilities of votes
        :param visualize:
        :return: a vedo object and the point
        """
        vtk_object = None
        point_sample = None

        while True:
            self.idx_votes += 1
            if self.votes is None:
                break
            idx_sample = self.votes[self.idx_votes][0]
            point_sample = self.np_pc_tested[idx_sample]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.c([200,200,200])
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                print(point_sample, " at ", angle_sample)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                break

        if visualize:
            vp = Plotter(verbose=0, title="Scores", bg="white", size=(1200, 800))
            vedo_file_env = load(self.file_mesh_env)
            vedo_file_env.backFaceCulling(True)
            vp.add(vedo_file_env)
            pts = Points(self.np_pc_tested, r=5)
            pts.cellColors(self.np_mapped_scores, cmap='jet_r', vmin=0, vmax=1)
            pts.addScalarBar(c='jet_r', nlabels=5, pos=(0.8, 0.25))
            vp.add(pts)
            vp.add(vtk_object)
            vp.show()
            vp.close()

        return vtk_object, point_sample

    def get_n_sample_clustered(self, min_score, n_samples, best_in_cluster=False,  visualize=False):
        """
          Get n samples from tested cluster_points where a minimum score is achieved (inclusive). Selection is performed used a cluster
        representation. If no enough point it choose all of them
        :param min_score: look for cluster_points with a minimum score
        :param n_samples: number of samples to extract
        :param best_in_cluster: if true the best is cluster is selected it not a vote system  is used
        :param visualize:
        :return:
        """
        selected_mapped_scores = self.np_mapped_scores[self.np_mapped_scores >= min_score]
        selected_tested_points = self.np_pc_tested[self.np_mapped_scores >= min_score]

        samples_point =[]
        samples_vedo_obj = []

        if len(selected_mapped_scores)>=n_samples:
            samples_vedo_obj, samples_point =  self.sample_n_by_k_means(selected_tested_points, selected_mapped_scores,
                                                                        n_samples, best_in_cluster, visualize)
        elif len(selected_mapped_scores)>0:
            samples_vedo_obj, samples_point =self.sample_all(selected_tested_points, visualize)




        return samples_vedo_obj, samples_point


    def sample_n_by_k_means(self, selected_tested_points, selected_mapped_scores, n_samples, best_in_cluster, visualize ):
        samples_point =[]
        samples_vedo_obj =[]
        kmeans = KMeans(init="random", n_clusters=n_samples, n_init=10, max_iter=300)
        kmeans.fit(selected_tested_points)
        distances = np.transpose([np.linalg.norm(selected_tested_points - c, axis=1) for c in kmeans.cluster_centers_])
        classes = np.argmin(distances, axis=1)

        for i in range(n_samples):
            cluster_scores = selected_mapped_scores[classes == i]
            cluster_points = selected_tested_points[classes == i]

            if best_in_cluster:
                idx_selected = random.choice(np.where(cluster_scores == np.amax(cluster_scores)))[0]
            else:
                cluster_votes = self.generate_votes(cluster_scores)
                if cluster_votes is None:
                    break
                idx_selected = cluster_votes[0][0]  # the most voted in cluster

            point_sample = cluster_points[idx_selected]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.c([200, 200, 200])
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                # print(angle_sample, " at ", point_sample)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                samples_point.append(point_sample)
                samples_vedo_obj.append(vtk_object)

        if visualize:
            vp = Plotter(verbose=0, title="Similarity", bg="white", size=(1200, 800))
            vedo_file_env = load(self.file_mesh_env)
            vedo_file_env.backFaceCulling(True)
            vp.add(vedo_file_env)
            pts = Points(self.np_pc_tested, r=5)
            pts.cellColors(self.np_mapped_scores, cmap='jet_r', vmin=0, vmax=1)
            pts.addScalarBar(c='jet_r', title="Similarity", nlabels=5, pos=(0.8, 0.25))
            vp.add(pts)
            for centroid in kmeans.cluster_centers_:
                vedo_centroid = vedo.Cone(centroid,r=.01, height=.03, c=[255, 255, 255])
                vp.add(vedo_centroid)
            colors = distinctipy.get_colors(n_samples)
            for i in range(n_samples):
                vp.add(samples_vedo_obj[i])
                r = np.max(distances[classes == i, i])
                vp.add(Sphere(pos=kmeans.cluster_centers_[i], r=r, c=colors[i], alpha=.2))
            vp.show()
            vp.close()

        return samples_vedo_obj, samples_point


    def sample_all(self, selected_tested_points, visualize ):
        samples_point =[]
        samples_vedo_obj =[]

        n_selected_points = len(selected_tested_points)

        for i in range(n_selected_points):

            point_sample = selected_tested_points[i]
            angle_sample = self.angle_with_best_score(x=point_sample[0], y=point_sample[1], z=point_sample[2])
            if angle_sample != -1:
                vtk_object = load(self.ply_obj_file)
                vtk_object.c([200, 200, 200])
                vtk_object.rotate(angle_sample, axis=(0, 0, 1), rad=True)
                # print(angle_sample, " at ", point_sample)
                vtk_object.pos(x=point_sample[0], y=point_sample[1], z=point_sample[2])
                samples_point.append(point_sample)
                samples_vedo_obj.append(vtk_object)

        if visualize:
            vp = Plotter(verbose=0, title="Similarity", bg="white", size=(1200, 800))
            vedo_file_env = load(self.file_mesh_env)
            vedo_file_env.backFaceCulling(True)
            vp.add(vedo_file_env)
            pts = Points(self.np_pc_tested, r=5)
            pts.cellColors(self.np_mapped_scores, cmap='jet_r', vmin=0, vmax=1)
            pts.addScalarBar(c='jet_r', title="Similarity", nlabels=5, pos=(0.8, 0.25))
            vp.add(pts)
            for i in range(n_selected_points):
                vp.add(samples_vedo_obj[i])
            vp.show()
            vp.close()

        return samples_vedo_obj, samples_point
