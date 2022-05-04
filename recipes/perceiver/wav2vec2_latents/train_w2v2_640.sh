#!/bin/bash
CFG=recipes/perceiver/wav2vec2_latents/train.yaml
GPU=3
SAVE_DIR=recipes/perceiver/wav2vec2_latents/results/train_w2v2_640_latents
export CUDA_VISIBLE_DEVICES=$GPU
cd ../../../

NUM_LATENTS=640

BATCH_SIZE=32
CLASSIFIER=perceiver_w2v2
CLUSTERIZE=0
EPOCHS=400
LR=0.0001
LR_GAMMA=0.98
OPTIMIZER=adamw
SEED=0

echo "Running Perceiver W2V2 experiment with $NUM_LATENTS latents..."
python train.py --config $CFG --batch_size $BATCH_SIZE --batch_size_dev $BATCH_SIZE --batch_size_test $BATCH_SIZE --lr $LR --lr_gamma $LR_GAMMA --optimizer $OPTIMIZER --run_dir $SAVE_DIR --classifier $CLASSIFIER --max_epochs $EPOCHS --prc_num_latents $NUM_LATENTS --clusterize_latents $CLUSTERIZE --seed $SEED
