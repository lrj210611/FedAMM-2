#!/bin/bash
time=$(date "+%m%d-%H%M")
python train.py \
--client_num 4 \
--c_rounds 1000 \
--round_per_train 100 \
--version ${time}_vesion \
--device_ids 0,1,2,3 \
--use_multiprocessing True \