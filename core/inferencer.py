# -*- coding: utf-8 -*-
"""统一推理入口：LLMInferencer 与模型可用性校验。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from core.adapters import OllamaAdapter, SGLangAdapter, VLLMAdapter
from core.adapters.base import BaseLLMAdapter
from core.config import CONFIG
from core.exceptions import (
    EngineNotInstalledError,
    InvalidParameterError,
    StructuredOutputNotSupportedError,
)

logger = logging.getLogger("core.inferencer")


class LLMInferencer:
    ENGINE_MAP = {"ollama": OllamaAdapter, "vllm": VLLMAdapter, "sglang": SGLangAdapter}

    def __init__(
        self,
        engine_type: Literal["ollama", "vllm", "sglang"],
        model_name: Optional[str] = None,
        *,
        ollama_base_url: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        vllm_local_model_path: Optional[str] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        sglang_base_url: Optional[str] = None,
    ):
        if engine_type not in self.ENGINE_MAP:
            raise InvalidParameterError(f"engine_type 仅允许 ollama/vllm/sglang，当前: {engine_type}")
        self.engine_type = engine_type
        self.model_name = model_name or CONFIG.get("default_model_name", "llama3.2")
        adapter_cls = self.ENGINE_MAP[engine_type]
        if engine_type == "ollama":
            url = ollama_base_url or (CONFIG.get("ollama") or {}).get("base_url", "http://localhost:11434")
            self._adapter: BaseLLMAdapter = adapter_cls(base_url=url)
        elif engine_type == "vllm":
            vllm_cfg = CONFIG.get("vllm") or {}
            url = vllm_base_url if vllm_base_url is not None else vllm_cfg.get("base_url")
            path = vllm_local_model_path if vllm_local_model_path is not None else vllm_cfg.get("local_model_path")
            self._adapter = adapter_cls(base_url=url, local_model_path=path, gpu_memory_utilization=vllm_gpu_memory_utilization)
        else:
            url = sglang_base_url or (CONFIG.get("sglang") or {}).get("base_url", "http://localhost:30000")
            self._adapter = adapter_cls(base_url=url)
        if not self._adapter.is_available():
            raise EngineNotInstalledError(f"引擎 '{engine_type}' 依赖未安装")
        self._adapter.check_service(self.model_name)
        logger.info("LLMInferencer 初始化成功: engine=%s, model=%s", engine_type, self.model_name)

    def _validate_common(self, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95) -> None:
        if not (0 <= temperature <= 2):
            raise InvalidParameterError("temperature 需在 [0, 2] 范围内")
        if max_tokens < 1:
            raise InvalidParameterError("max_tokens 需为正整数")
        if not (0 <= top_p <= 1):
            raise InvalidParameterError("top_p 需在 [0, 1] 范围内")

    def generate(self, prompt: str, *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.generate(prompt, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def chat(self, messages: List[Dict[str, str]], *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.chat(messages, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def structured_generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> Union[Dict[str, Any], str]:
        if self.engine_type != "sglang":
            raise StructuredOutputNotSupportedError("structured_generate 仅支持 SGLang")
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.structured_generate(prompt, schema, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)


def validate_model_usable(inferencer: LLMInferencer, max_tokens: int = 5) -> bool:
    """验证已启动的模型是否可用：执行一次短文本生成，成功返回 True，异常返回 False。"""
    try:
        inferencer.generate("hi", max_tokens=max_tokens)
        return True
    except Exception as e:
        logger.warning("验证模型可用性失败: %s", e)
        return False
