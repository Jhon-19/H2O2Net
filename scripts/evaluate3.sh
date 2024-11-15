#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=03 \
--device_id=0 \
--mode=test \
--corpus_type=WebNLG \
--restore_file=best \
--use_link \
