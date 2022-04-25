#!/bin/bash
CFG=recipes/perceiver/wav2vec2_latents/baseline_vs_w2v2_base.yaml
GPU=0
SAVE_DIR=recipes/perceiver/wa2vec2_latents/results/baseline_vs_w2v2_base
export CUDA_VISIBLE_DEVICES=$GPU
cd ../../../

echo "Running Perceiver base frozen experiment..."
CLASSIFIER=perceiver
RUN_DIR=$SAVE_DIR/baseline_frozen
FREEZE=1
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_freeze_latents $FREEZE --seed $SEED
done

echo "Running Perceiver W2V2 frozen experiment..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_freeze_latents $FREEZE --seed $SEED
done

echo "Running Perceiver base unfrozen experiment..."
CLASSIFIER=perceiver
RUN_DIR=$SAVE_DIR/baseline_unfrozen
FREEZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_freeze_latents $FREEZE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_freeze_latents $FREEZE --seed $SEED
done
