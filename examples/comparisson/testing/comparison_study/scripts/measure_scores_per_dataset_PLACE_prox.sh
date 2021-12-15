#!/bin/bash

export HDF5_USE_FILE_LOCKING='FALSE'

#conda activate keras_gpu
export PYTHONPATH=/home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE:/home/alexa/Abel/DATA/git_repositories/iTpy:/home/alexa/Abel/DATA/git_repositories/iTpyClearance:/home/alexa/Abel/DATA/git_repositories/PLACE:/home/alexa/Abel/DATA/git_repositories/smplx:\$PYTHONPATH

DEVICE=1




#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene 17DRP5sb8fy-bedroom 
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene 17DRP5sb8fy-livingroom 
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene 17DRP5sb8fy-familyroomlounge 
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene X7HyMhZNoso-livingroom_0_16 
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene sKLMLpTHeUy-familyname_0_1 
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene zsNo4HB9uLZ-bedroom0_0
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene zsNo4HB9uLZ-livingroom0_13



CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene MPH16
CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene MPH1Library
CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene N0SittingBooth
CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene N3OpenArea




#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene apartment_1
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene frl_apartment_0
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene hotel_0
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene office_2
#CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/04_measure_PLACE_scores_1_calculating.py --scene room_0


