#!/bin/bash

# # Configuration variables
# GPU_IDS=(0)

# MASTER_PORT=29411

# # Arrays of tasks and generation lengths
# TASKS=("gsm8k")
# GEN_LENGTHS=(256)

# # Set GPU IDs from command line if provided
# if [ $# -gt 0 ]; then
#   # Clear default GPU list and add provided GPUs
#   GPU_IDS=()
#   for arg in "$@"; do
#     GPU_IDS+=("$arg")
#   done
# fi

# GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
# NUM_GPUS=${#GPU_IDS[@]}
# echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

# for task in "${TASKS[@]}"; do
#   for gen_length in "${GEN_LENGTHS[@]}"; do
#     # Set batch size based on generation length
#     if [ "$gen_length" -eq 512 ]; then
#       batch_size=4
#     else
#       batch_size=8
#     fi
    
#     echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
#     CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
#       --nproc_per_node $NUM_GPUS \
#       --master_port $MASTER_PORT \
#       eval.py \
#       --dataset $task \
#       --batch_size $batch_size \
#       --gen_length $gen_length \
#       --output_dir "eval_results" \
#       --model_path "/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/merged_models/checkpoint-44800_merged"
#       --checkpoint_path "/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/merged_models/checkpoint-44800_merged"
#   done
# done


# echo "All evaluations completed!"


##########


# Configuration variables
GPU_IDS=(0)

MASTER_PORT=29413

# Arrays of tasks and generation lengths
TASKS=("gsm8k")
GEN_LENGTHS=(256)

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
      --output_dir "eval_results" \
      --model_path "/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base" 
  done
done


echo "All evaluations completed!"
