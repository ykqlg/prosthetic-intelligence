#!/bin/bash

python -u main.py \
--test_only \
--repeat_num 3 \
--winstep 0.01 \
--numcep 13 \
--winlen 0.025 \
--nfilt 26 \
--nfft 512 \
--ceplifter 22 \
--train_label_file ./data/white_cup_user1_label.csv \
--test_label_file \
./data/yellow_cup_label.csv \
./data/white_cup_user2_label.csv \
./data/box_label.csv \
./data/velcro_label.csv \
./data/badminton_label.csv \
./data/ping_pong_label.csv \
./data/toilet_roll_label.csv \
--val_label_file ./data/yellow_cup_label.csv \
--test_size 0.4 \
--random_state 42 \
--fs 1330 \
--forward 0.45 \
--backward 1.0
