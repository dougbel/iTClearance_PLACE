import gc
import json
import os
import sys
import time
import traceback

import trimesh

import numpy as np
from mpi4py import MPI
from vedo import write, merge
import vedo

from it_clearance.preprocessing.bubble_filler import BubbleFiller
from it_clearance.testing.envirotester import EnviroTesterClearance
from mpi_master_slave import Slave


class SlaveNoFilledEnviroTester(Slave):
    """
     A slave process extends Slave class, overrides the 'do_work' method
     and calls 'Slave.run'. The Master will do the rest
     """
    def __init__(self, dataset_scans_path, work_directory, config_directory):
        super().__init__()
        self.dataset_scans_path = dataset_scans_path
        self.work_directory = work_directory
        self.descriptors_repository = os.path.join(config_directory, "descriptors_repository")
        self.json_exec_file_dir = os.path.join(config_directory, "json_execution")

        self.rank = MPI.COMM_WORLD.Get_rank()
        self.name = MPI.Get_processor_name()

    def do_work(self, data):
        dataset  = data[0]
        scene  = data[1]
        interaction  = data[2]
        try:
            # print('  Slave %s rank %d sampling scan "%s"' % (name, rank, scan))
            gc.collect()
            env_file = os.path.join(self.dataset_scans_path, dataset, "scenes",  scene+'.ply')
            env_file_filled = os.path.join(self.dataset_scans_path, dataset, "scenes_filled",  scene+'.ply')
            tri_mesh_env = trimesh.load_mesh(env_file)


            test_points_file = os.path.join(self.work_directory,"samples", scene, "sample_points.npy")
            test_normals_file = os.path.join(self.work_directory, "samples", scene, "sample_normals.npy")
            sampling_data_file = os.path.join(self.work_directory, "samples", scene, "sampling_data.json")

            np_test_points = np.load(test_points_file)
            np_env_normals = np.load(test_normals_file)

            json_conf_execution_file = os.path.join(self.json_exec_file_dir, f"single_testing_{interaction}.json")

            tester = EnviroTesterClearance(self.descriptors_repository, json_conf_execution_file)
            time_start = time.time()
            results_it_test = tester.start_full_test(tri_mesh_env, tri_mesh_env, np_test_points, np_env_normals)
            time_finish = time.time()
            time_exe_it_test = time_finish - time_start

            # ##################################################################################################################
            # SAVING output

            for idx_aff in range(len(tester.affordances)):
                # iter = tester.affordances[idx_aff][0] + "_"+tester.affordances[idx_aff][1]
                it_name = tester.affordances[idx_aff][0]

                # creating output directory
                output_dir = os.path.join(self.work_directory, "no_filled_env_test", scene, it_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                test_data = {}
                test_data['execution_time_it_test'] = time_exe_it_test
                test_data['sampling_data_file'] = sampling_data_file
                with open(sampling_data_file) as jsonfile:
                    test_data['sampling_data']  = json.load(jsonfile)
                test_data['tester_info'] = tester.configuration_data
                test_data['directory_of_trainings'] = tester.working_path
                test_data['file_env'] = env_file
                test_data['file_env_filled'] = env_file_filled
                with open(os.path.join(output_dir, 'test_data.json'), 'w') as outfile:
                    json.dump(test_data, outfile, indent=4)

                # it scores
                filename = os.path.join(output_dir, "test_scores.csv")  # "%s/test_scores.csv" % output_dir
                df_by_interaction = results_it_test.loc[(results_it_test.interaction == it_name)]
                df_by_interaction.to_csv(filename)

            return True, data, self.name, self.rank, ""
        except:
            return False, data, self.name, self.rank, traceback.format_exc()
