# -*- coding: utf-8 -*-
"""VLLM 引擎适配器。"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import httpx

from core.adapters.base import BaseLLMAdapter
from core.config import CONFIG
from core.exceptions import EngineNotInstalledError, EngineNotRunningError, InvalidParameterError

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

logger = logging.getLogger("core.adapters.vllm")


def _is_path_allowed(path: str) -> bool:
    """检查路径是否位于允许的模型目录内，防止路径遍历。"""
    from core.config import get_platform_hf_dir, get_platform_models_dir
    abs_path = os.path.abspath(os.path.expanduser(path))
    allowed_roots = [
        get_platform_models_dir(),
        get_platform_hf_dir(),
        os.getcwd(),
    ]
    vllm_cfg = CONFIG.get("vllm") or {}
    local_path = vllm_cfg.get("local_model_path")
    if local_path:
        allowed_roots.append(os.path.abspath(os.path.expanduser(local_path)))
    for root in allowed_roots:
        # 使用 os.path.commonpath 比较规范化后的绝对路径
        try:
            common = os.path.commonpath([abs_path, root])
            if common == root:
                return True
        except ValueError:
            continue
    return False


def _vllm_resolve_path(model_name: str, local_model_path: Optional[str]) -> str:
    def _expand(s: str) -> str:
        return os.path.expanduser(s) if (s.startswith("~") or s.startswith("/")) else s
    if local_model_path:
        resolved = _expand(local_model_path)
        if not _is_path_allowed(resolved):
            raise InvalidParameterError(f"vLLM 本地模型路径不在允许目录内: {resolved}")
        return resolved
    vllm_cfg = CONFIG.get("vllm") or {}
    if vllm_cfg.get("local_model_path"):
        resolved = _expand(vllm_cfg["local_model_path"])
        if not _is_path_allowed(resolved):
            raise InvalidParameterError(f"vLLM 本地模型路径不在允许目录内: {resolved}")
        return resolved
    raw = (vllm_cfg.get("model_aliases") or {}).get(model_name, model_name)
    resolved = _expand(raw)
    # 统一解析为绝对路径后校验；HF repo ID 等无斜杠的字符串通常为合法标识
    abs_resolved = os.path.abspath(resolved)
    if not _is_path_allowed(abs_resolved):
        raise InvalidParameterError(f"vLLM 模型路径不在允许目录内: {resolved}")
    return resolved


class VLLMAdapter(BaseLLMAdapter):
    def __init__(self, base_url: Optional[str] = None, local_model_path: Optional[str] = None, gpu_memory_utilization: Optional[float] = None, quantization: Optional[str] = None):
        self.base_url = base_url
        self.local_model_path = local_model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._quantization = quantization
        self._llm: Any = None
        self._llm_lock = threading.Lock()

    @property
    def engine_type(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        if self.base_url:
            try:
                from urllib.parse import urljoin
                with httpx.Client(timeout=2.0) as c:
                    return c.get(urljoin(self.base_url.rstrip("/") + "/", "health")).status_code == 200
            except Exception:
                return False
        return VLLM_AVAILABLE

    def _get_llm(self, model_name: str) -> Any:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm。请执行: pip install vllm")
        if self.base_url:
            return None
        path = _vllm_resolve_path(model_name, self.local_model_path)
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    vllm_cfg = CONFIG.get("vllm") or {}
                    gpu_util = self._gpu_memory_utilization if self._gpu_memory_utilization is not None else vllm_cfg.get("gpu_memory_utilization")
                    kwargs: Dict[str, Any] = {"model": path, "trust_remote_code": True}
                    if gpu_util is not None:
                        kwargs["gpu_memory_utilization"] = float(gpu_util)
                    if self._quantization is not None:
                        kwargs["quantization"] = self._quantization
                    self._llm = LLM(**kwargs)
        return self._llm

    def check_service(self, model_name: str) -> None:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm")
        if self.base_url:
            try:
                with httpx.Client(timeout=5.0) as c:
                    # 健壮地拼接 health 端点 URL
                    from urllib.parse import urljoin
                    health_url = urljoin(self.base_url.rstrip("/") + "/", "health")
                    r = c.get(health_url)
                    if r.status_code != 200:
                        raise EngineNotRunningError(f"VLLM 服务未就绪: {self.base_url}")
            except httpx.RequestError as e:
                raise EngineNotRunningError(f"无法连接 VLLM: {e}")
        else:
            self._get_llm(model_name)

    def _messages_to_prompt(self, messages: List[Dict[str, str]], model_name: str) -> str:
        # 仅在本地模式尝试使用 transformers 的 chat_template
        if not self.base_url:
            try:
                from transformers import AutoTokenizer
                path = _vllm_resolve_path(model_name, self.local_model_path)
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        # 回退到简易拼接
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
        if self.base_url:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{self.base_url}/v1/completions", json={"model": model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p})
                r.raise_for_status()
                choices = r.json().get("choices", [])
                return (choices[0].get("text", "") if choices else "").strip()
        llm = self._get_llm(model_name)
        with self._llm_lock:
            sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
            outs = llm.generate([prompt], sampling)
        return (outs[0].outputs[0].text if outs and outs[0].outputs else "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        prompt = self._messages_to_prompt(messages, model_name)
        return self.generate(prompt, model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)
