#!/bin/bash

export HDF5_USE_FILE_LOCKING='FALSE'

#conda activate keras_gpu
export PYTHONPATH=/home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE:/home/alexa/Abel/DATA/git_repositories/iTpy:/home/alexa/Abel/DATA/git_repositories/iTpyClearance:/home/alexa/Abel/DATA/git_repositories/PLACE:/home/alexa/Abel/DATA/git_repositories/smplx:\$PYTHONPATH

DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE python /home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE/examples/comparisson/testing/comparison_study/03_measure_scores_per_dataset_1_calculating.py

