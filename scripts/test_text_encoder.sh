#!/usr/bin/env bash
source /nas/home/meryem/anaconda3/bin/activate yourenvname
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
python ../model/encoders/text_encoder_pytorch.py