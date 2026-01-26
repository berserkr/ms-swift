# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedModel

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
from ..model_arch import ModelArch
from ..utils import AttnImpl

logger = get_logger()

class GraniteLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger.info(
            '[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
            'be really slow!')
        return super().get_model(model_dir, *args, **kwargs)

    def _update_attn_impl(self, config):
        use_flash_attn = AttnImpl.to_use_flash_attn(self.attn_impl, 'auto')
        config.use_flash_attn = use_flash_attn


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
        GraniteLoader,
        template=TemplateType.granite,
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
        GraniteLoader,
        template=TemplateType.granite,
        architectures=['GraniteForCausalLM'],
        model_arch=ModelArch.granite_dense))