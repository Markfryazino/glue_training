#!/bin/bash

for task in "sst2" "rte" "cola"
do
    for seed in 1 2 3
    do
        python3 logreg_glue.py --task $task --seed $seed --log_file results.csv --use_wandb \
                               --wandb_entity broccoliman --wandb_project tinkoff_qualification
        for model in "microsoft/deberta-v3-base" "xlm-roberta-base"
        do
            python3 transformer_glue.py --task $task --model $model --seed $seed --epochs 3 \
                                        --log_file results.csv --use_wandb \
                                        --wandb_entity broccoliman --wandb_project tinkoff_qualification
        done
    done
done