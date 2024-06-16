#!/bin/bash

python -u main.py \
--test_only \
--repeat_num 5 \
--winstep 0.01 \
--numcep 10 \
--winlen 0.025 \
--nfilt 11 \
--nfft 256 \
--no-dynamic \
--train_label_file ./data/white_cup_user1_label.csv \
--test_label_file ./data/white_cup_user2_label.csv \
--test_size 0.3 \
--random_state 42 \
--fs 1330 \
--forward 0.02 \
--backward 0.8
