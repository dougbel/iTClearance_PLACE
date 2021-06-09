import itertools
import json
import os
import time
import logging
from shutil import copyfile

import pandas as pd

from mpi_master_slave import Master, WorkQueue

class MasterRoutines(object):
    """
    This is my application that has a lot of work to do so it gives work to do
    to its slaves until all the work is done
    THIS ROUTINE WORKS ONLY ONE TYPE AT A TIME
    """

    def __init__(self, slaves, work_directory, follow_up_column):
        # when creating the Master we tell it what slaves it can handle
        self.master = Master(slaves)
        # WorkQueue is a convenient class that run slaves on a tasks queue
        self.work_queue = WorkQueue(self.master)
        self.work_directory = work_directory
        self.follow_up_file_name = 'follow_up_process.csv'
        self.follow_up_column = follow_up_column

        logging_file = os.path.join(work_directory, 'process_' + follow_up_column + '.log')
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

        follow_up_file = os.path.join(self.work_directory, self.follow_up_file_name)
        follow_up_data = pd.read_csv(follow_up_file, index_col=[0,1,2])

        if not self.follow_up_column in follow_up_data.columns:
            follow_up_data[self.follow_up_column] = False

        num_total_task = follow_up_data.index.size
        pending_tasks = list(follow_up_data[follow_up_data[self.follow_up_column] == False].index)
        num_completed_task = follow_up_data[follow_up_data[self.follow_up_column] == True].index.size

        logging.info(
            'STARTING TASKS: total %d, done %d, pendings %d' % (num_total_task, num_completed_task, len(pending_tasks)))

        """
        This is the core of my application, keep starting slaves
        as long as there is work to do
        """
        #
        # let's prepare our work queue. This can be built at initialization time
        # but it can also be added later as more work become available
        #
        for data in pending_tasks:
            # 'data' will be passed to the slave and can be anything
            self.work_queue.add_work(data=data)

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
                    follow_up_data.at[data, self.follow_up_column] = True
                    num_completed_task += 1
                    logging.info('%d/%d, name %s slave rank %d processing %s on "%s", success: %s' % (
                        num_completed_task, num_total_task, name, rank, self.follow_up_column, ', '.join(data),
                        str.upper(str(done))))
                else:
                    logging.error('name %s slave rank %d processing %s on "%s": \n%s' % (
                        name, rank, self.follow_up_column, ', '.join(data) , exception_info))

            if not num_completed_task % 3 and num_completed_task > 0:
                copyfile(follow_up_file, follow_up_file + "_backup")
                follow_up_data.to_csv(follow_up_file)

            # sleep some time
            time.sleep(0.3)

        follow_up_data.to_csv(follow_up_file)