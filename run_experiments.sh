#!/bin/bash

[ -e results.csv ] && rm results.csv
for task in "sst2" "rte" "cola"
do
    for seed in 1 2 3
    do
        python3 logreg_glue.py --task $task --seed $seed --log_file results.csv "$@"
        for model in "distilbert-base-cased" "distilroberta-base"
        do
            python3 transformer_glue.py --task $task --model $model --seed $seed --epochs 2 \
                                        --log_file results.csv "$@"
        done
    done
done