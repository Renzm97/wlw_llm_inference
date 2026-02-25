# -*- coding: utf-8 -*-
"""Ollama 引擎适配器。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from core.adapters.base import BaseLLMAdapter
from core.exceptions import EngineNotRunningError, ModelNotFoundError

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

logger = logging.getLogger("core.adapters.ollama")


class OllamaAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @property
    def engine_type(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        if OLLAMA_AVAILABLE:
            return True
        try:
            with httpx.Client(timeout=2.0) as c:
                return c.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def check_service(self, model_name: str) -> None:
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{self.base_url}/api/tags")
                logger.debug("Ollama check_service %s -> status=%s", self.base_url, r.status_code)
                if r.status_code != 200:
                    raise EngineNotRunningError(f"Ollama 服务未就绪: {self.base_url} (status={r.status_code})")
                data = r.json()
                models_list = data.get("models") or []
                names = [(m.get("model") or m.get("name") or "") for m in models_list]
                logger.info("Ollama 已拉取模型列表 base_url=%s 数量=%s 列表=%s", self.base_url, len(names), names)
                if model_name not in names and not any(
                    n and (n == model_name or n.startswith(model_name + ":") or model_name in n) for n in names
                ):
                    raise ModelNotFoundError(f"模型 '{model_name}' 不在 Ollama 已拉取列表中。可用: {names}。")
        except httpx.RequestError as e:
            logger.warning("Ollama check_service 连接失败 base_url=%s: %s", self.base_url, e)
            raise EngineNotRunningError(f"无法连接 Ollama: {e}")

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p},
        }
        logger.debug("Ollama generate base_url=%s model=%s", self.base_url, model_name)
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p},
        }
        logger.debug("Ollama chat base_url=%s model=%s", self.base_url, model_name)
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()
