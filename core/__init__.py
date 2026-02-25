# -*- coding: utf-8 -*-
"""
推理核心包：配置、异常、引擎适配器、统一推理入口与模型下载。
对外保持与原 inference_core 兼容的导出。
"""

from core.config import (
    BUILTIN_MODELS,
    CONFIG,
    MODELS_CATALOG,
    ensure_model_downloaded,
    get_model_variant,
    get_models_catalog,
    get_platform_hf_dir,
    get_platform_models_dir,
)
from core.exceptions import (
    EngineNotInstalledError,
    EngineNotRunningError,
    InvalidParameterError,
    LLMInferenceError,
    ModelNotFoundError,
    StructuredOutputNotSupportedError,
)
from core.inferencer import LLMInferencer, validate_model_usable

# 兼容旧代码中的 _ensure_model_downloaded / _get_platform_*
_ensure_model_downloaded = ensure_model_downloaded
_get_platform_models_dir = get_platform_models_dir
_get_platform_hf_dir = get_platform_hf_dir

__all__ = [
    "BUILTIN_MODELS",
    "CONFIG",
    "MODELS_CATALOG",
    "get_models_catalog",
    "get_model_variant",
    "ensure_model_downloaded",
    "_ensure_model_downloaded",
    "get_platform_models_dir",
    "get_platform_hf_dir",
    "LLMInferencer",
    "validate_model_usable",
    "LLMInferenceError",
    "EngineNotInstalledError",
    "EngineNotRunningError",
    "ModelNotFoundError",
    "InvalidParameterError",
    "StructuredOutputNotSupportedError",
]
