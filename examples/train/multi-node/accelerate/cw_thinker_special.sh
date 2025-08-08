#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=16
#SBATCH --job-name=7b-p4-lc-128k-swift-ot3-ignore_think_1e-5_spectok
#SBATCH --ntasks-per-node=1  #<--must be 1 for torchrun / override for others like mpi
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144 
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/7b-p4-lc-128k-swift-ot3-ignore_think_1e-5_spectok-out.%j.log" 
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/7b-p4-lc-128k-swift-ot3-ignore_think_1e-5_spectok-err.%j.log" 
####SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0
#SBATCH --segment=2
####SBATCH --segment=9 # 9 18 <-- currently commented out and experimenting how it impacts 256 node job
####SBATCH --exclusive # <-- currently commented out and experimenting how it impacts 256 node job

####run this command on slurm login node: 
#### sbatch -N 16 /mnt/home/bobcalio/ai-coreweave/dolomite_engine/scripts/cw-gb200/pretrain-120b.sbatch <config>

#### Variables
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8 #2 # has to be 2 for 30b, 1 for 120b
#SEQLEN=32768
#SEQLEN=40960
#SEQLEN=24576
SEQLEN=16384
#65536
#SEQLEN=4096
#LR=9e-05
LR=5e-06
CLIP=1.0

WARMUP_RATIO=0.1
LR_SCHEDULE_TYPE=linear

MIX_NAME="7b_s_v2_1_args_spectok"
NUM_EPOCHS=1

timestamp=$(date +"%Y%m%d_%H%M%S")
ID="e4_${WARMUP_RATIO}_${LR_SCHEDULE_TYPE}-clipping_${CLIP}"

MODEL_BASE_PATH=/mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/edited
NAME=7b-p4-lc-128k
SHORT_NAME=7b-p4-lc-128k

#MODEL_BASE_PATH=/mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/30b-p3-v1/transformers_compatible
#NAME=global_step92500
#SHORT_NAME=30b-p3-v1

MODEL_PATH=$MODEL_BASE_PATH/$NAME
#DATA_MIX_PATH=/mnt/vast/proj/checkpoints/bathen/datasets/sft/tokenized/tulu3_sft_jsonl/40k
#DATA_MIX_PATH=/mnt/vast/proj/datasets/sft-datasets/g4-stage1-tokenized/mixtures/
DATA_MIX_PATH=/mnt/vast/proj/datasets/sft-datasets/fusion
OUTPUT_BASE_PATH=/mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/sft
OUTPUT_MODEL_NAME="${SHORT_NAME}-${MIX_NAME}-${SEQLEN}-${GRADIENT_ACCUMULATION_STEPS}-${PER_DEVICE_TRAIN_BATCH_SIZE}-${NUM_EPOCHS}-${LR}-${CLIP}-${ID}"
OUTPUT_MODEL_PATH="${OUTPUT_BASE_PATH}/${OUTPUT_MODEL_NAME}-model"

CHAT_TEMPLATE_PATH=/mnt/home/bathen/src/github.com/granite-chat-template/granite_4.0/chat_template.jinja2

. ~/.bashrc
source ~/run.env

min_bw=$( echo "140" | bc -l)
sleep="300s"

#weights and biases
export WANDB_BASE_URL=https://wandbai.draco.res.ibm.com
export WANDB_ENTITY=bathen
export WANDB_PROJECT=granite-4-models-sft
#export WANDB_RUN_ID=sft-test
export WANDB_DISABLE_CODE=1
export WANDB_DISABLE_GIT=1
export WANDB__SERVICE_WAIT=300

: "${PREFLIGHT_TEST:=0}"
: "${CLEANUP_TEMP_DIR:=0}"

PYXIS_DEFAULTS=( '--no-container-mount-home' '--no-container-remap-root')

#container_image="/mnt/vast/squash/open-instruct-g4-v3.sqsh"
#container_image="/mnt/vast/squash/open-instruct-g4-tf4520.sqsh"

container_mounts="/mnt:/mnt"
container_image="/mnt/vast/squash/swift_v2.sqsh"
LOG=/mnt/vast/proj/checkpoints/bathen/logs/${SHORT_NAME}_${SLURM_JOBID}.log

# from MLPerf team -- need top review 
#. ${HOME}/ai-coreweave/dolomite_engine/scripts/cw-gb200/config_common.sh

#default nccl vars handled in .nccl.conf
export TOKENIZERS_PARALLELISM=false 
export NCCL_SOCKET_IFNAME=eth0
#export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export PMIX_MCA_gds='^ds12'
export NCCL_MIN_CTAS=32
export NCCL_NET_GDR_C2C=1
export NCCL_WORK_FIFO_DEPTH=1048576
##
export NCCL_TIMEOUT_WAIT_SEC=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch If set to 1, aborting NCCL communicator and tearing down process upon error.
export PYTHONUNBUFFERED=TRUE #<print right away 
export OMP_NUM_THREADS=64 #<--can adjust, but for now leave to remove warning 
#export OMP_DYNAMIC=true

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
#export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export JOB_ID=${SLURM_JOBID}

export NCCL_DEBUG=WARN #INFO 
export NCCL_DEBUG_SUBSYS=BOOTSTRAP,INIT,NET,ENV
export NCCL_DEBUG_FILE=${NCCL_LOGS_PATH}/NCCL_DEBUG_FILE.%h.txt

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch If set to 1, aborting NCCL communicator and tearing down process upon error.

export TORCH_CUDA_ARCH_LIST="Blackwell" # 12.0+PTX
export CUTE_ARCH_LDSM_SM100A_ENABLED=1 
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TORCHINDUCTOR_REORDER_FOR_PEAK_MEMORY=1

mkdir -p /tmp/$USER/triton
export TRITON_HOME=/tmp/$USER/triton
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST" >> $LOG
#save hostlist for replay / debug if needed 
# echo $SLURM_JOB_NODELIST > $run_dir/hostfile-${SLURM_JOB_ID}.txt
#setup some srun args 
SRUN_ARGS="--kill-on-bad-exit=1  \
            --container-image=${container_image}  \
            --container-mounts=${container_mounts}  \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/ms-swift
            "
echo $SRUN_ARGS >> $LOG

echo `date` : ${SLURM_JOBID} >> $LOG

export DISTRIBUTED_ARGS="--mixed_precision bf16 \
    --num_machines ${SLURM_JOB_NUM_NODES} \
    --num_processes ${WORLD_SIZE} \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --rdzv_backend c10d \
    "
echo $DISTRIBUTED_ARGS >> $LOG

#export SCRIPT_ARGS="--model /mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/lc-ckpts/7b-p4-lc-128k/hf \
#/mnt/vast/proj/checkpoints/bathen/datasets/sft/OpenThoughts3-1.2M/ot3.messages.jsonl
#/mnt/vast/proj/datasets/sft-datasets/jsonl/mixes/hermes3-tools-rag-ot3.jsonl
export SCRIPT_ARGS="--model  /mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/edited/7b-p4-lc-128k \
    --train_type full \
    --dataset /mnt/vast/proj/checkpoints/bathen/datasets/sft/OpenThoughts3-1.2M/ot3.messages.jsonl \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --loss_scale ignore_think \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 1 \
    --max_length 32768 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32 \
    --dataset_num_proc 32 \
    --save_total_limit 5 \
    --save_only_model true \
    --output_dir /mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/sft/7b-p4-lc-128k-swift-ot3-ignore_think_1e-5_spectok \
    --attn_impl flash_attn \
    --use_chat_template true \
    
    "
echo $SCRIPT_ARGS >> $LOG

CONFIG=examples/train/multi-node/accelerate/fsdp_accelerate.yaml
SCRIPT=swift/cli/sft.py

echo "CUDA DEVICES: ${CUDA_VISIBLE_DEVICES}" >> $LOG
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ${DISTRIBUTED_ARGS} --config_file ${CONFIG} ${SCRIPT} ${SCRIPT_ARGS}"

echo "*********************** START ****************************" >> $LOG
echo $CMD >> $LOG

srun ${SRUN_ARGS} bash -c "${CMD}"
