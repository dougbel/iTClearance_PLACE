
export PYTHONPATH=/home/dougbel/Documents/git_repositories/my_projects/iT_comparing/iTClearance_PLACE:/home/dougbel/Documents/git_repositories/my_projects/iT_development/iTpy:/home/dougbel/Documents/git_repositories/my_projects/iT_generator_data/si:/home/dougbel/Documents/git_repositories/my_projects/iT_development/iTpyClearance:/home/dougbel/Documents/git_repositories/third_part/libraries/mpi-master-slave:/home/dougbel/Documents/UoB/5th_semestre/to_test/place_comparisson/PLACE:\$PYTHONPATH

cd /home/dougbel/Documents/git_repositories/my_projects/iT_comparing/iTClearance_PLACE/examples/comparisson/testing/comparison_study/piloting


SRCS=/home/dougbel/Documents/git_repositories/my_projects/iT_comparing/iTClearance_PLACE

python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_1_5.csv --dataset replica_v1 --weight_collision 1 --weight_loss_contact 1.5 --weight_loss_vposer 0.10
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_1_5.csv --dataset mp3d --weight_collision 1 --weight_loss_contact 1.5 --weight_loss_vposer 0.10
## execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_1_5.csv
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_1_5.csv
#
#
## exactly as in the PLACE paper
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_01_contact_0_5.csv --dataset replica_v1 --weight_collision 0.01 --weight_loss_contact 0.5 --weight_loss_vposer 0.10
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_01_contact_0_5.csv --dataset mp3d --weight_collision 0.01 --weight_loss_contact 0.5 --weight_loss_vposer 0.10
## execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_01_contact_0_5.csv
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_01_contact_0_5.csv
#
#
## experiment number 3
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_2_contact_0_5.csv --dataset replica_v1 --weight_collision 0.2 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_2_contact_0_5.csv --dataset mp3d --weight_collision 0.2 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_0_2_contact_0_5.csv --measure_it_clearance_metrics False
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_0_2_contact_0_5.csv --measure_it_clearance_metrics False


# experiment number 4
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_5_contact_1_0.csv --dataset replica_v1 --weight_collision 1.5 --weight_loss_contact 1.0 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_5_contact_1_0.csv --dataset mp3d --weight_collision 1.5 --weight_loss_contact 1.0 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
# execution for generating latex tables
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_5_contact_1_0.csv --measure_it_clearance_metrics False
python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_5_contact_1_0.csv --measure_it_clearance_metrics False
#
#
## experiment number 5
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_0_5.csv --dataset replica_v1 --weight_collision 1.0 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_1_calculating[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_0_5.csv --dataset mp3d --weight_collision 1.0 --weight_loss_contact 0.5 --weight_loss_vposer 0.10 --measure_it_clearance_metrics False
## execution for generating latex tables
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_replica_coll_1_0_contact_0_5.csv --measure_it_clearance_metrics False
#python $SRCS/examples/comparisson/testing/comparison_study/piloting/03_measure_scores_per_dataset_2_summarize[PILOTING].py --follow_up_file_name piloting_itClearance_mp3d_coll_1_0_contact_0_5.csv --measure_it_clearance_metrics Falses