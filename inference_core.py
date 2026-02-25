#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容层：从 core 包重新导出，保持旧代码「from inference_core import ...」可用。
新代码请直接使用: from core import ...
"""

from core import (
    BUILTIN_MODELS,
    CONFIG,
    LLMInferencer,
    _ensure_model_downloaded,
    validate_model_usable,
)
from core.exceptions import (
    EngineNotInstalledError,
    EngineNotRunningError,
    InvalidParameterError,
    LLMInferenceError,
    ModelNotFoundError,
    StructuredOutputNotSupportedError,
)

__all__ = [
    "BUILTIN_MODELS",
    "CONFIG",
    "LLMInferencer",
    "validate_model_usable",
    "_ensure_model_downloaded",
    "LLMInferenceError",
    "EngineNotInstalledError",
    "EngineNotRunningError",
    "ModelNotFoundError",
    "InvalidParameterError",
    "StructuredOutputNotSupportedError",
]
