#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/train.py \
--ex_index=4e \
--epoch_num=100 \
--device_id=0 \
--corpus_type=WebNLG-star \
--use_link \
