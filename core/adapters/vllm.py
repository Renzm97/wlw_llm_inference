# -*- coding: utf-8 -*-
"""VLLM 引擎适配器。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from core.adapters.base import BaseLLMAdapter
from core.config import CONFIG
from core.exceptions import EngineNotInstalledError, EngineNotRunningError

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

logger = logging.getLogger("core.adapters.vllm")


def _vllm_resolve_path(model_name: str, local_model_path: Optional[str]) -> str:
    def _expand(s: str) -> str:
        return os.path.expanduser(s) if (s.startswith("~") or s.startswith("/")) else s
    if local_model_path:
        return _expand(local_model_path)
    vllm_cfg = CONFIG.get("vllm") or {}
    if vllm_cfg.get("local_model_path"):
        return _expand(vllm_cfg["local_model_path"])
    raw = (vllm_cfg.get("model_aliases") or {}).get(model_name, model_name)
    return _expand(raw)


class VLLMAdapter(BaseLLMAdapter):
    def __init__(self, base_url: Optional[str] = None, local_model_path: Optional[str] = None, gpu_memory_utilization: Optional[float] = None):
        self.base_url = base_url
        self.local_model_path = local_model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None

    @property
    def engine_type(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        return VLLM_AVAILABLE

    def _get_llm(self, model_name: str) -> Any:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm。请执行: pip install vllm")
        if self.base_url:
            return None
        path = _vllm_resolve_path(model_name, self.local_model_path)
        if self._llm is None:
            vllm_cfg = CONFIG.get("vllm") or {}
            gpu_util = self._gpu_memory_utilization if self._gpu_memory_utilization is not None else vllm_cfg.get("gpu_memory_utilization")
            kwargs: Dict[str, Any] = {"model": path, "trust_remote_code": True}
            if gpu_util is not None:
                kwargs["gpu_memory_utilization"] = float(gpu_util)
            self._llm = LLM(**kwargs)
        return self._llm

    def check_service(self, model_name: str) -> None:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm")
        if self.base_url:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(f"{self.base_url.replace('/v1', '')}/health")
                    if r.status_code != 200:
                        raise EngineNotRunningError(f"VLLM 服务未就绪: {self.base_url}")
            except httpx.RequestError as e:
                raise EngineNotRunningError(f"无法连接 VLLM: {e}")
        else:
            self._get_llm(model_name)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
        parts.append("Assistant: ")
        return "\n".join(parts)

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        if self.base_url:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{self.base_url}/completions", json={"model": model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p})
                r.raise_for_status()
                choices = r.json().get("choices", [])
                return (choices[0].get("text", "") if choices else "").strip()
        llm = self._get_llm(model_name)
        outs = llm.generate([prompt], sampling)
        return (outs[0].outputs[0].text if outs and outs[0].outputs else "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)
