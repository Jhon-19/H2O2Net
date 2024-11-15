#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=01 \
--device_id=0 \
--mode=test \
--corpus_type=NYT \
--restore_file=best \
--use_link \
