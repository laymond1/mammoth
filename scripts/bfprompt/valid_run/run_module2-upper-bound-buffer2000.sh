#!/bin/bash

# CIL CONFIG
NAME="Online BFPrompt with module2 (upper-bound) and M=2000"
NOTE="si-blurry-bfprompt-module2-upper-bound-M=2000"  # Short description of the experiment.
SCENARIO="si-blurry"  # Mode configuration
BUFFER_SIZE=2000
# model config
MODEL="online-bfprompt"
GT_KEY_VALUE=1
PROMPT_PRED=0

# Define the list of datasets
DATASETS=("seq-cifar100-224" "seq-imagenet-r" "seq-cars196" "seq-cub200" "seq-eurosat-rgb" "seq-resisc45" "seq-chestx" "seq-isic" "seq-cropdisease")

# Loop over each dataset
for DATASET in "${DATASETS[@]}"; do
    # Define default model, optimizer, and scheduler configurations
    if [ "$DATASET" == "seq-cifar100-224" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=1000 F_EVAL_PERIOD=10000
        BATCHSIZE=64; LR=5e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=10

    elif [ "$DATASET" == "seq-tinyimg-224" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=2000 F_EVAL_PERIOD=20000
        BATCHSIZE=64; LR=5e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=10

    elif [ "$DATASET" == "seq-imagenet-r" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=1000 F_EVAL_PERIOD=6000
        BATCHSIZE=64; LR=5e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=10
    
    elif [ "$DATASET" == "seq-cars196" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=100 F_EVAL_PERIOD=1000
        BATCHSIZE=64; LR=1e-3 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=10

    elif [ "$DATASET" == "seq-cub200" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=100 F_EVAL_PERIOD=1000
        BATCHSIZE=64; LR=1e-3 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=10

    elif [ "$DATASET" == "seq-eurosat-rgb" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=270 F_EVAL_PERIOD=2700
        BATCHSIZE=64; LR=1e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=5

    elif [ "$DATASET" == "seq-resisc45" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=500 F_EVAL_PERIOD=5000
        BATCHSIZE=64; LR=5e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=9
    
    elif [ "$DATASET" == "seq-chestx" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=100 F_EVAL_PERIOD=1000
        BATCHSIZE=64; LR=1e-3 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=2

    elif [ "$DATASET" == "seq-isic" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=100 F_EVAL_PERIOD=500
        BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=3

    elif [ "$DATASET" == "seq-cropdisease" ]; then
        ONLINE_ITER=3 EVAL_PERIOD=1000 F_EVAL_PERIOD=10000
        BATCHSIZE=64; LR=1e-4 OPT_NAME="adam" SCHED_NAME="default"
        N_TASKS=7

    else
        echo "Undefined setting for dataset: $DATASET"
        exit 1
    fi

    # Loop over random seeds
    for seed in 1; do
        echo "Running on dataset: $DATASET with seed: $seed"

        CUDA_VISIBLE_DEVICES=1 python utils/online_main.py \
        --notes $NOTE --seed $seed \
        --online_scenario $SCENARIO \
        --model $MODEL --gt_key_value $GT_KEY_VALUE --prompt_prediction $PROMPT_PRED \
        --dataset $DATASET \
        --n_tasks $N_TASKS \
        --optimizer $OPT_NAME \
        --lr $LR --lr_scheduler $SCHED_NAME \
        --batch_size $BATCHSIZE --online_iter $ONLINE_ITER \
        --eval_period $EVAL_PERIOD --f_eval_period $F_EVAL_PERIOD \
        --num_workers 4 --log_path "./results/" \
        --wandb_entity "laymond1" --wandb_project "Hyperparam Tuning on Si-Blurry" \
        --validation 0.2 --buffer_size $BUFFER_SIZE
    done
done
