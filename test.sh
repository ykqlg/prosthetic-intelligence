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
--no-dynamic \
--train_label_file ./data/train_label_file.csv \
--test_label_file ./data/test_label_file.csv \
--test_size 0.2 \
--random_state 42 \
--fs 1330 \
--forward 0.45 \
--backward 0.8
