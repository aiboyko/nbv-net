#!/bin/bash

path_to_data="/Users/dmitriismirnov/dataset/"
vol_train="vol_train.npy"
lbl_train="lbl_train.npy"
vol_test="vol_test.npy"
lbl_test="lbl_test.npy"

echo "Prepare vol and lbl data:"
#python3 prepare_data.py --dataset_folder $path_to_data

echo "Train data:"
python3 train.py \
        --dataset_folder $path_to_data \
        --file_vol_train $vol_train \
        --file_lbl_train $lbl_train \
        --file_vol_test $vol_test \
        --file_lbl_test $lbl_test \


echo "Visualize output:"




#python3 -m cnbv --dataset=dataset --continuous_validation
#python3 -m cnbv --dataset=dataset --continuous_validation --net=our_net
