#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/train.py \
--ex_index=332 \
--epoch_num=100 \
--device_id=0 \
--corpus_type=WebNLG \
--use_link \

