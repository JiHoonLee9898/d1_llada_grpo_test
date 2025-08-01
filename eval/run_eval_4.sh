#!/bin/bash

##########
# Configuration variables
GPU_IDS=(0)

MASTER_PORT=29417

# Arrays of tasks and generation lengths
TASKS=("gsm8k")
GEN_LENGTHS=(256)
DIFF_STEPS=(128)
MODEL_PATHS=("/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base")
DECODINGS=("low_confidence" "top_k_margin")

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    for diffusion_step in "${DIFF_STEPS[@]}"; do
      for model in "${MODEL_PATHS[@]}"; do
        for decoding in "${DECODINGS[@]}"; do
          # Set batch size based on generation length
          if [ "$gen_length" -eq 512 ]; then
          batch_size=4
          else
          batch_size=8
          fi
          
          echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
          
          CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
          --nproc_per_node $NUM_GPUS \
          --master_port $MASTER_PORT \
          eval.py \
          --dataset $task \
          --batch_size $batch_size \
          --gen_length $gen_length \
          --diffusion_steps $diffusion_step \
          --decoding $decoding \
          --output_dir "/home/work/jihoon_wombat_storage/JIHOON/d1_jihoon/eval/eval_results_20250730" \
          --model_path $model \
          --seed 42
        done
      done
    done
  done
done


echo "All evaluations completed!"
