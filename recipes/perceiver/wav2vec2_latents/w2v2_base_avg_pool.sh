#!/bin/bash
CFG=recipes/perceiver/wav2vec2_latents/baseline_vs_w2v2_base_clusters.yaml
GPU=2
SAVE_DIR=recipes/perceiver/wav2vec2_latents/results/w2v2_base_avg_pool
LATENT_MODE=avg_pool
export CUDA_VISIBLE_DEVICES=$GPU
cd ../../../

NUM_LATENTS=20
echo "Running Perceiver W2V2 frozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

NUM_LATENTS=40
echo "Running Perceiver W2V2 frozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

NUM_LATENTS=80
echo "Running Perceiver W2V2 frozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

NUM_LATENTS=160
echo "Running Perceiver W2V2 frozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

NUM_LATENTS=320
echo "Running Perceiver W2V2 frozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_frozen
FREEZE=1
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done

echo "Running Perceiver W2V2 unfrozen experiment with $NUM_LATENTS latents..."
CLASSIFIER=perceiver_w2v2
RUN_DIR=$SAVE_DIR/w2v2_base_unfrozen
FREEZE=0
CLUSTERIZE=0
for SEED in {1..10}
do
    python train.py --config $CFG --run_dir $RUN_DIR --classifier $CLASSIFIER --prc_num_latents $NUM_LATENTS --prc_freeze_latents $FREEZE --clusterize_latents $CLUSTERIZE --latent_process_mode $LATENT_MODE --seed $SEED
done
