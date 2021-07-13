#!/bin/bash

export HDF5_USE_FILE_LOCKING='FALSE'

#conda activate keras_gpu
export PYTHONPATH=/home/alexa/Abel/projects/04_place_execution/iTClearance_PLACE:/home/alexa/Abel/DATA/git_repositories/iTpy:/home/alexa/Abel/DATA/git_repositories/iTpyClearance:/home/alexa/Abel/DATA/git_repositories/PLACE:/home/alexa/Abel/DATA/git_repositories/smplx:\$PYTHONPATH


DEVICE=1




CUDA_VISIBLE_DEVICES=$DEVICE python iTClearance_PLACE/examples/comparisson/testing/03_test_place_AUTO_selected_points.py  --base_dir /home/alexa/Abel/DATA/PLACE_comparisson

