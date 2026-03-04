# 35GiB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model /mnt/vast/proj/checkpoints/bathen/models/base/Qwen2.5-3B \
    --tuner_type full \
    --dataset /mnt/vast/proj/datasets/sft-datasets/jsonl/preview_mix/granite-4.0-sft-datasets-1216/phase1_mix_1216_v1_mt_only.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --use_liger_kernel true \
    --output_dir output \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16
