# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_batch

import re
import json

@dataclass
class GraniteTemplateMeta(TemplateMeta):
    """
    This final template will have space for chat template separation, but suffix will have no new line as it is eos
    """
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
        ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_text|>\n'])
    prefix : Prompt = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|end_of_text|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|start_of_role|>system<|end_of_role|>{{SYSTEM}}<|end_of_text|>\n'])
    agent_template: str = 'granite_agentic'


@dataclass
class GraniteTemplateMetaV1(TemplateMeta):
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
        ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_text|>\n'])
    prefix : Prompt = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>\n'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|end_of_text|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|start_of_role|>system<|end_of_role|>{{SYSTEM}}<|end_of_text|>\n'])
    agent_template: str = 'granite_agentic'


@dataclass
class GraniteTemplateMetaV2(TemplateMeta):
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
        ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_text|>\n'])
    prefix : Prompt = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|end_of_text|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|start_of_role|>system<|end_of_role|>{{SYSTEM}}<|end_of_text|>\n'])
    agent_template: str = 'granite_agentic'


@dataclass
class GraniteTemplateMetaV3(TemplateMeta):
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
        ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_text|>'])
    prefix : Prompt = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|start_of_role|>system<|end_of_role|>{{SYSTEM}}<|end_of_text|>\n'])
    agent_template: str = 'granite_agentic'


@dataclass
class GraniteTemplateMetaFinal(TemplateMeta):
    """
    This final template will have space for chat template separation, but suffix will have no new line as it is eos
    """
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'
        ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_text|>\n'])
    prefix : Prompt = field(default_factory=lambda: [])
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|end_of_text|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|start_of_role|>system<|end_of_role|>{{SYSTEM}}<|end_of_text|>\n'])
    agent_template: str = 'granite_agentic'

register_template(GraniteTemplateMeta(LLMTemplateType.granite))
