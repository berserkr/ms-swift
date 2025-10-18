export MODELSCOPE_CACHE=/mnt/vast/proj/checkpoints/bathen/cache
export MEGATRON_LM_PATH=/mnt/home/bathen/src/github.com/Megatron-LM

MODEL=/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-30B-A3B-Base
MCORE_MODEL=/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-30B-A3B-Base-mcore

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model $MODEL \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir $MCORE_MODEL \
    --test_convert_precision true