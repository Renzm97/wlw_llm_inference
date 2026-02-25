#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容层：从 core、services、api 重新导出，保持旧代码与「python llm_inference.py --serve」可用。
新代码请使用: python main.py --serve 或 from api import create_app；from core import ...
"""

from __future__ import annotations

# 推理核心（与 inference_core 一致）
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

# 服务层：运行实例
from services import (
    get_running_inferencer,
    start_model_impl,
    stop_model_impl,
    user_facing_start_error,
)
from services.instances import RUNNING_INSTANCES, _running_lock

# 兼容旧命名
_start_model_impl = start_model_impl
_stop_model_impl = stop_model_impl
_get_running_inferencer = get_running_inferencer
_user_facing_start_error = user_facing_start_error

# API 应用（供 uvicorn 或外部挂载）
try:
    from api import FASTAPI_AVAILABLE, create_app
    app = create_app() if FASTAPI_AVAILABLE else None
except ImportError:
    FASTAPI_AVAILABLE = False
    create_app = None
    app = None

# 测试入口（兼容旧命令行）
from main import run_api_tests, run_core_tests

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
    "get_running_inferencer",
    "start_model_impl",
    "stop_model_impl",
    "user_facing_start_error",
    "RUNNING_INSTANCES",
    "_running_lock",
    "create_app",
    "app",
    "FASTAPI_AVAILABLE",
    "run_core_tests",
    "run_api_tests",
]

if __name__ == "__main__":
    from main import main
    main()
