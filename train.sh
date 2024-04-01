#!/bin/bash

python -u main.py \
--repeat_num 3 \
--winstep 0.01 \
--numcep 13 \
--nfilt 26 \
--nfft 512 \
--ceplifter 22 \
--no-concat \
--no-dynamic \
--label_file_path ./label_file.csv \
--test_size 0.2 \
--random_state 42 \
--fs 1330 \
--forward 0.45 \
--backward 0.8
