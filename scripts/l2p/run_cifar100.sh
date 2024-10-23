#!/bin/bash

# CIL CONFIG
NOTE="si-blurry-l2p"  # Short description of the experiment.
MODEL="online-l2p"
SCENARIO="si-blurry"  # Mode configuration
DATASET="seq-cifar100-224"  # Dataset options: cifar100, tinyimg, imagenet-r
N_TASKS=5
N=50
M=10
USE_AMP="--use_amp"

# Define default model, optimizer, and scheduler configurations
if [ "$DATASET" == "seq-cifar100-224" ]; then
    MEM_SIZE=0 ONLINE_ITER=3 EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "seq-tinyimg-224" ]; then
    MEM_SIZE=0 ONLINE_ITER=3 EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "seq-imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3 EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi

# Loop over random seeds
for seed in 1 2 3 4 5;
do
    CUDA_VISIBLE_DEVICES=0 python utils/online_main.py \
    --notes $NOTE --seed $seed \
    --online_scenario $SCENARIO \
    --model $MODEL \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --optimizer $OPT_NAME \
    --lr $LR --lr_scheduler $SCHED_NAME \
    --batch_size $BATCHSIZE --online_iter $ONLINE_ITER \
    --eval_period $EVAL_PERIOD \
    --num_workers 8 --log_path "./results/" \
    --wandb_entity "laymond1" --wandb_project "Si-Blurry Scenario" \
    $USE_AMP
done
