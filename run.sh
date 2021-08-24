#!/bin/bash

source arne-env/bin/activate
CUDA_VISIBLE_DEVICES=-1 python main.py  --transformer \
                --d_model 512 \
                --h 10 \
                --n_layer 6 \
                --d_att 64 \
                --dropout_transformer 0.1 \
                --lr 0.00005 \
                --lr_decay 0.75 \
                --lr_decay_threshold 0.6
deactivate