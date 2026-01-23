# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Tuple, Type

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_device_count, get_dist_setting, get_env_args, get_logger
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..patcher import patch_fixed_device, patch_get_input_embeddings, patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, get_model_tokenizer_reward_model,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl, ModelInfo, use_submodel_func

logger = get_logger()
dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}


def get_model_tokenizer_granite(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if model_info.torch_dtype is not None:
        k_true = dtype_mapping[model_info.torch_dtype]
        for k in dtype_mapping.values():
            setattr(model_config, k, k == k_true)

    quantization_config = model_kwargs.get('quantization_config')
    if not isinstance(quantization_config, BitsAndBytesConfig):
        # not bnb quant
        model_config.torch_dtype = None
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
    model_config.use_flash_attn = use_flash_attn
    kwargs['model_config'] = model_config
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.eod_id
    kwargs['tokenizer'] = tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    try:
        # fix mp+ddp bug
        #NOTE: Is this needed? (Luis)
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer


#TODO: Check if it should be get_model_tokenizer_with_flash_attn ?
register_model(
    ModelMeta(
        LLMModelType.granite,
        [
            # granite
            ModelGroup([
                # moe
                Model('IBM/granite-4-tiny', 'IBM/granite-4-tiny'),
                Model('IBM/granite-4-small', 'IBM/granite-4-small'),
                Model('IBM/granite-4-medium', 'IBM/granite-4-medium'),
            ]),
        ],
        TemplateType.granite,
        get_model_tokenizer_granite,
        architectures=['GraniteMoeHybridForCausalLM'],
        model_arch=ModelArch.granite))

register_model(
    ModelMeta(
        LLMModelType.granite_dense,
        [
            # granite
            ModelGroup([
                # dense
                Model('IBM/granite-4-3b', 'IBM/granite-4-3b'),
            ]),
        ],
        TemplateType.granite,
        get_model_tokenizer_granite,
        architectures=['GraniteForCausalLM'],
        model_arch=ModelArch.granite_dense))