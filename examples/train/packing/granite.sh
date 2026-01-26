# Env: 4 * A100
# https://github.com/modelscope/ms-swift/blob/main/examples/megatron/long_text.sh
# Max Length: 16K
# GPU Memory: 4 * 42GB, Training Speed 10s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model /mnt/vast/proj/checkpoints/bathen/models/base/granite-4.0-3b-base-prerelease-killington-final-hybridclass \
    --tuner_type full \
    --dataset /mnt/vast/proj/datasets/sft-datasets/jsonl/preview_mix/granite-4.0-sft-datasets-1216/phase1_mix_1216_v1.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir /mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/sft/granite-4.0-3b-sft-test \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --agent_template granite_agentic \
    --loss_scale granite \
    --use_chat_template true \

