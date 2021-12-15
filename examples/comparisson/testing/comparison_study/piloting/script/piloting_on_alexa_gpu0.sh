#!/bin/bash

export HDF5_USE_FILE_LOCKING='FALSE'

SRCS=/home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE

#conda activate keras_gpu
export PYTHONPATH=$SRCS:/home/alexa/Abel/DATA/git_repositories/iTpy:/home/alexa/Abel/DATA/git_repositories/iTpyClearance:/home/alexa/Abel/DATA/git_repositories/PLACE:/home/alexa/Abel/DATA/git_repositories/smplx:\$PYTHONPATH

DEVICE=0


cd $SRCS/examples/comparisson/testing/comparison_study/piloting




#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_1_5.csv --dataset replica_v1 --weight_collision 1 --weight_loss_contact 1.5 --weight_loss_vposer 0.10
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_1_5.csv --dataset mp3d --weight_collision 1 --weight_loss_contact 1.5 --weight_loss_vposer 0.10
# execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_1_5.csv
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_1_5.csv


# exactly as in the PLACE paper
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_01_contact_0_5.csv --dataset replica_v1 --weight_collision 0.01 --weight_loss_contact 0.5 --weight_loss_vposer 0.10
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_01_contact_0_5.csv --dataset mp3d --weight_collision 0.01 --weight_loss_contact 0.5 --weight_loss_vposer 0.10
# execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_01_contact_0_5.csv
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_01_contact_0_5.csv


# experiment number 3
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_2_contact_0_5.csv --dataset replica_v1 --weight_collision 0.2 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_2_contact_0_5.csv --dataset mp3d --weight_collision 0.2 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_2_contact_0_5.csv --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_2_contact_0_5.csv --measure_it_clearance_metrics False


# experiment number 4
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_5_contact_1_0.csv --dataset replica_v1 --weight_collision 1.5 --weight_loss_contact 1.0 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_5_contact_1_0.csv --dataset mp3d --weight_collision 1.5 --weight_loss_contact 1.0 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_5_contact_1_0.csv --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_5_contact_1_0.csv --measure_it_clearance_metrics False


# experiment number 5
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_0_5.csv --dataset replica_v1 --weight_collision 1.0 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_0_5.csv --dataset mp3d --weight_collision 1.0 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_0_5.csv --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_0_5.csv --measure_it_clearance_metrics Falses


# experiment number r1
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_15_contact_0_5.csv --dataset replica_v1 --weight_collision 0.15 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_15_contact_0_5.csv --measure_it_clearance_metrics False


# experiment number r2
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_10_contact_0_5.csv --dataset replica_v1 --weight_collision 0.10 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_10_contact_0_5.csv --measure_it_clearance_metrics False