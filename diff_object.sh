#!/bin/bash

python -u main.py \
--test_only \
--repeat_num 5 \
--winstep 0.01 \
--numcep 13 \
--winlen 0.025 \
--nfilt 10 \
--nfft 512 \
--ceplifter 22 \
--train_label_file ./data/white_cup_user1_label.csv \
--tests_label_file \
./data/white_cup_user2_label.csv \
./data/yellow_cup_label.csv \
./data/box_label.csv \
./data/velcro_label.csv \
./data/badminton_label.csv \
./data/ping_pong_label.csv \
./data/toilet_roll_label.csv \
--val_label_file ./data/yellow_cup_label.csv \
--test_size 0.3 \
--random_state 42 \
--fs 1330 \
--forward 0.05 \
--backward 0.8
