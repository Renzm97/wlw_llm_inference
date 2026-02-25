# -*- coding: utf-8 -*-
"""SGLang 引擎适配器。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from core.adapters.base import BaseLLMAdapter
from core.exceptions import EngineNotInstalledError, EngineNotRunningError

try:
    import sglang
    from sglang import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    sglang = None
    Engine = None
    SGLANG_AVAILABLE = False

logger = logging.getLogger("core.adapters.sglang")


class SGLangAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url.rstrip("/")
        self._engine: Any = None

    @property
    def engine_type(self) -> str:
        return "sglang"

    def is_available(self) -> bool:
        return SGLANG_AVAILABLE

    def check_service(self, model_name: str) -> None:
        if not SGLANG_AVAILABLE:
            raise EngineNotInstalledError("未安装 sglang")
        try:
            with httpx.Client(timeout=5.0) as c:
                if c.get(f"{self.base_url}/get_model_info").status_code != 200:
                    raise EngineNotRunningError(f"SGLang 服务未就绪: {self.base_url}")
        except httpx.RequestError as e:
            raise EngineNotRunningError(f"无法连接 SGLang: {e}")

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        payload = {"text": prompt, "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}}
        with httpx.Client(timeout=120.0) as c:
            r = c.post(f"{self.base_url}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("text", "") or data.get("generated_text", "") or "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        parts.append("assistant: ")
        return self.generate("\n".join(parts), model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def structured_generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> Union[Dict[str, Any], str]:
        self.check_service(model_name)
        schema_hint = f"\n请严格按以下 JSON 结构返回：\n{json.dumps(schema, ensure_ascii=False)}\n" if schema else ""
        full_prompt = f"{prompt}{schema_hint}\n请只输出合法 JSON，不要其他文字。"
        raw = self.generate(full_prompt, model_name=model_name, temperature=max(0.1, temperature), max_tokens=max_tokens, top_p=top_p, **kwargs)
        try:
            raw = raw.strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
