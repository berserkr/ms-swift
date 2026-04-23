#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=32
#SBATCH --job-name=granite-8b-thinking-glm
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/granite-8b-thinking-glm-out.%j.log"
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/granite-8b-thinking-glm-err.%j.log"
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0

. ~/.bashrc
source ~/run.env

export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export NCCL_TIMEOUT_WAIT_SEC=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=64

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export NCCL_DEBUG=WARN

export WANDB_ENTITY=ai-models
export WANDB_PROJECT=granite-8b-thinking-sft
export WANDB_DISABLE_CODE=1
export WANDB_DISABLE_GIT=1

container_mounts="/mnt:/mnt"
container_image="/mnt/vast/squash/swift_v3_scattermoe.sqsh"

SRUN_ARGS="--kill-on-bad-exit=1 \
            --container-image=${container_image} \
            --container-mounts=${container_mounts} \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/ms-swift"

export DISTRIBUTED_ARGS="--mixed_precision bf16 \
    --num_machines ${SLURM_JOB_NUM_NODES} \
    --num_processes ${WORLD_SIZE} \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --rdzv_backend c10d"

export MODELSCOPE_CACHE=/mnt/vast/proj/checkpoints/bathen/cache

# GLM data paths — all 4 splits
GLM_DATA="/mnt/vast/proj/checkpoints/bathen/datasets/sft/GLM-5.1-Reasoning-1M-Cleaned/main_openai.jsonl"
GLM_DATA="${GLM_DATA} /mnt/vast/proj/checkpoints/bathen/datasets/sft/GLM-5.1-Reasoning-1M-Cleaned/Math_openai.jsonl"
GLM_DATA="${GLM_DATA} /mnt/vast/proj/checkpoints/bathen/datasets/sft/GLM-5.1-Reasoning-1M-Cleaned/PHD-Science_openai.jsonl"
GLM_DATA="${GLM_DATA} /mnt/vast/proj/checkpoints/bathen/datasets/sft/GLM-5.1-Reasoning-1M-Cleaned/Multilingual-STEM_openai.jsonl"

export SCRIPT_ARGS="--model /mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-8b-base-special \
    --template_type granite_thinking \
    --train_type full \
    --dataset ${GLM_DATA} \
    --torch_dtype bfloat16 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 1 \
    --max_length 32768 \
    --gradient_checkpointing true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir /mnt/vast/proj/checkpoints/bathen/models/swift/granite-8b-thinking-glm \
    --attn_impl flash_attn \
    --use_liger_kernel true"

CONFIG=examples/train/multi-node/accelerate/fsdp_accelerate.yaml
SCRIPT=swift/cli/sft.py

CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ${DISTRIBUTED_ARGS} --config_file ${CONFIG} ${SCRIPT} ${SCRIPT_ARGS}"

echo "$(date) Starting: granite-8b-thinking-glm"
echo "CMD: ${CMD}"

srun ${SRUN_ARGS} bash -c "${CMD}"
rc=$?
echo "$(date) Finished with rc=${rc}"
exit $rc
