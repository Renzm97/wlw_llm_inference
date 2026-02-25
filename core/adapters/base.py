# -*- coding: utf-8 -*-
"""LLM 引擎适配器基类。"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union


class BaseLLMAdapter(abc.ABC):
    @property
    @abc.abstractmethod
    def engine_type(self) -> str:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    @abc.abstractmethod
    def check_service(self, model_name: str) -> None:
        pass

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        pass

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        pass

    def structured_generate(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], str]:
        return self.generate(
            prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )
