import argparse
import json
import logging
import time
import gc
import traceback
import os

import trimesh
from mpi4py import MPI
from mpi_master_slave import Master, WorkQueue, Slave
import numpy as np

class MasterRoutines(object):
    """
    This is my application that has a lot of work to do so it gives work to do
    to its slaves until all the work is done
    THIS ROUTINE WORKS ONLY ONE TYPE AT A TIME
    """

    def __init__(self, slaves, scans_dir, scene_name, grid_dim, output_dir):
        # when creating the Master we tell it what slaves it can handle
        self.master = Master(slaves)
        # WorkQueue is a convenient class that run slaves on a tasks queue
        self.work_queue = WorkQueue(self.master)
        self.scans_dir = scans_dir
        self.scene_name = scene_name
        self.grid_dim = grid_dim
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging_file = os.path.join(output_dir, f'process_sdf_{scene_name}.log')
        logging.basicConfig(filename=logging_file, filemode='a', level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

        # Creating multiple scenes produces some harmless error messages, to AVOID
        logging.getLogger('pyembree').disabled = True

    def terminate_slaves(self):
        """
        Call this to make all slaves exit their run loop
        """
        self.master.terminate_slaves()

    def run(self):

        """
        This is the core of my application, keep starting slaves
        as long as there is work to do
        """
        #
        # let's prepare our work queue. This can be built at initialization time
        # but it can also be added later as more work become available
        #

        mesh_decimated = trimesh.load(os.path.join(self.scans_dir, self.scene_name + ".ply"))

        bb_padded_vertices = mesh_decimated.bounding_box.vertices * 1.7


        np_min = np.array(bb_padded_vertices.min(axis=0))
        np_max = np.array(bb_padded_vertices.max(axis=0))
        step_x, step_y, step_z = (np_max - np_min) / self.grid_dim
        init_x = step_x / 2 + np_min[0]
        init_y = step_y / 2 + np_min[1]
        init_z = step_z / 2 + np_min[2]

        points = np.empty((self.grid_dim, self.grid_dim, self.grid_dim, 3))

        for xi in range(self.grid_dim):
            for yi in range(self.grid_dim):
                for zi in range(self.grid_dim):
                    points[xi, yi, zi] = np.array([init_x + xi * step_x, init_y + yi * step_y, init_z + zi * step_z])
                # 'data' will be passed to the slave and can be anything
                self.work_queue.add_work(data=[xi,yi,points[xi, yi, :]])

        sdf_data = np.empty((self.grid_dim, self.grid_dim, self.grid_dim))

        #
        # Keeep starting slaves as long as there is work to do
        #
        while not self.work_queue.done():

            #
            # give more work to do to each idle slave (if any)
            #
            self.work_queue.do_work()

            #
            # reclaim returned data from completed slaves
            #
            for slave_return_data in self.work_queue.get_completed_work():
                done, data, name, rank, exception_info = slave_return_data
                if done:
                    sdf_data[data[0], data[1],:] = data[2]
                    logging.info(f'sdf calculated in x: {data[0]} , y: {data[1]}, name {name} slave rank {rank} processed, success: {done}')
                else:
                    logging.error(f'CALCULATING SDF VALUES in x: {data[0]} , y: {data[1]}, name {name} slave rank {rank}')


            # sleep some time
            time.sleep(0.3)


        # Saving results
        sdf_data_reshaped = sdf_data.reshape(self.grid_dim * self.grid_dim * self.grid_dim)
        np.save(os.path.join(self.output_dir, self.scene_name + "_sdf.npy"), sdf_data_reshaped)
        dictionary = {
            "max": list(np_max),
            "dim": self.grid_dim,
            "min": list(np_min)
        }
        with open(os.path.join(self.output_dir, self.scene_name + ".json"), "w") as outfile:
            json.dump(dictionary, outfile)





class SlaveEnviroTester(Slave):
    """
     A slave process extends Slave class, overrides the 'do_work' method
     and calls 'Slave.run'. The Master will do the rest
     """
    def __init__(self, scans_dir, scene ):
        super().__init__()
        self.scans_dir = scans_dir
        self.scene = scene
        self.mesh_decimated = trimesh.load(os.path.join(self.scans_dir, self.scene+".ply"))

        self.rank = MPI.COMM_WORLD.Get_rank()
        self.name = MPI.Get_processor_name()

    def do_work(self, data):
        xi  = data[0]
        yi  = data[1]
        partial_voxel_points  = data[2]
        try:
            # print('  Slave %s rank %d sampling scan "%s"' % (name, rank, scan))
            gc.collect()

            sdf_values = np.empty(partial_voxel_points.shape[0])
            (closest_points_in_env, norms, id_triangle) = self.mesh_decimated.nearest.on_surface(partial_voxel_points)

            for i in range(len(id_triangle)):
                v_p = partial_voxel_points[i]
                s_p = closest_points_in_env[i]
                s_n = self.mesh_decimated.face_normals[id_triangle[i]]
                sign = -1 if np.dot((v_p - s_p), s_n) < 0 else 1
                sdf_values[i] = sign * norms[i]

            data[2] = sdf_values

            return True, data, self.name, self.rank, ""
        except:
            return False, data, self.name, self.rank, traceback.format_exc()





parser = argparse.ArgumentParser()
parser.add_argument('--scans_dir', required=True, help='Path to decimated meshes')
parser.add_argument('--scene', required=True, help='scene')
parser.add_argument('--grid_dim', required=True, help='Dimesion of grid used')
parser.add_argument('--output_dir', required=True, help='Dimesion of grid used')
opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    name = MPI.Get_processor_name()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # activate it with debug purposes
    # mpiexec -n 2 python comparisson/sdf_generation/02_sdf_calculation_MPI_following_prox.py --scans_dir /media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/replica_v1/scenes_downsampled  --scene apartment_1 --grid_dim 10 --output_dir /media/dougbel/Tezcatlipoca/PLACE_trainings/datasets/replica_v1/sdf_tmp
    import pydevd_pycharm
    port_mapping = [44563, 41831]
    pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    print(os.getpid())

    print('***************************************************************************')
    print('I am  %s rank %d (total %d)' % (name, rank, size))
    print('***************************************************************************')

    if rank == 0:  # Master

        app = MasterRoutines(slaves=range(1, size), scans_dir=opt.scans_dir, scene_name=opt.scene, grid_dim=int(opt.grid_dim), output_dir=opt.output_dir)
        app.run()
        app.terminate_slaves()

    else:  # Any slave
        SlaveEnviroTester(scans_dir=opt.scans_dir, scene=opt.scene).run()

    print('Task completed')