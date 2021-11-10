import argparse

from mpi4py import MPI


from mpi_prox.master import MasterRoutines
from mpi_prox.slaves.slave_test_env import SlaveEnviroTester

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_scans_path', required=True, help='Path to ScanNet dataset')
parser.add_argument('--work_directory', required=True, help='Path to work_directory folder')
parser.add_argument('--config_directory', required=True, help='Path to descriptors and json config executions')
parser.add_argument('--use_filled_env', default=True, help='Path to descriptors and json config executions')
opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    name = MPI.Get_processor_name()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # activate it with debug purposes
    # mpiexec -n 2 python examples/comparisson/testing/ablation_study/01_testing_on_no_filled_env.py --dataset_scans_path /media/dougbel/Tezcatlipoca/PLACE_trainings/datasets --work_directory /media/dougbel/Tezcatlipoca/PLACE_trainings/test  --config_directory /media/dougbel/Tezcatlipoca/PLACE_trainings/config
    import pydevd_pycharm
    port_mapping = [33059, 39981]
    pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
    import os
    print(os.getpid())

    print('***************************************************************************')
    print('I am  %s rank %d (total %d)' % (name, rank, size))
    print('***************************************************************************')

    if rank == 0:  # Master

        app = MasterRoutines(slaves=range(1, size), work_directory=opt.work_directory, follow_up_column="no_filled_env_tested")
        app.run()
        app.terminate_slaves()

    else:  # Any slave
        SlaveEnviroTester(dataset_scans_path=opt.dataset_scans_path, work_directory=opt.work_directory,
                          config_directory=opt.config_directory, use_filled_env=opt.use_filled_env).run()

    print('Task completed (rank %d)' % (rank))
