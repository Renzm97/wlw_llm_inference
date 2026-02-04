#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量 LLM 推理核心模块 + FastAPI 接口层
对标 Xinference 基础推理能力，统一集成 Ollama / VLLM / SGLang 三大后端。

========== 快速开始指南 ==========
1. 依赖安装（必装）:
   pip install pydantic fastapi uvicorn requests httpx

2. 各引擎可选依赖:
   - Ollama: pip install ollama 或使用内置 httpx 调用本地 API
   - VLLM:   pip install vllm   (需 GPU/CUDA)
   - SGLang: pip install sglang (需 GPU/CUDA)

3. 环境准备:
   - Ollama: 先安装并启动 Ollama 服务，拉取模型如 llama3.2
   - VLLM:   本地需启动 vllm serve，或配置 base_url 远程调用
   - SGLang: 本地需启动 sglang 服务

4. 运行方式:
   - 仅运行测试: python llm_inference.py
   - 启动 API 服务: python llm_inference.py --serve [--host 0.0.0.0] [--port 8000]

5. 使用示例见文件末尾「使用示例」注释块与测试用例。
"""

# ---------- 依赖声明（与 requirements.txt 对应） ----------
# 必装: pydantic, fastapi, uvicorn, requests, httpx
# 可选: ollama (Ollama), vllm (VLLM), sglang (SGLang)

from __future__ import annotations

import abc
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

# 必装依赖直接导入
import httpx

# 软导入：未安装的引擎仅在使用时抛异常
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    vllm = None
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

try:
    import sglang
    from sglang import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    sglang = None
    Engine = None
    SGLANG_AVAILABLE = False


# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_inference")


# ---------- 自定义异常 ----------
class LLMInferenceError(Exception):
    """推理模块基础异常。"""


class EngineNotInstalledError(LLMInferenceError):
    """引擎对应依赖未安装。"""


class EngineNotRunningError(LLMInferenceError):
    """引擎服务未启动（如 Ollama/VLLM/SGLang 未运行）。"""


class ModelNotFoundError(LLMInferenceError):
    """指定模型不存在或不可用。"""


class InvalidParameterError(LLMInferenceError):
    """参数不合法（如 temperature 超出范围）。"""


class StructuredOutputNotSupportedError(LLMInferenceError):
    """当前引擎不支持结构化输出（仅 SGLang 支持）。"""


# ---------- 引擎适配基类 ----------
class BaseLLMAdapter(abc.ABC):
    """LLM 引擎适配基类，定义 generate / chat / structured_generate 统一抽象接口。"""

    @property
    @abc.abstractmethod
    def engine_type(self) -> str:
        """返回引擎类型标识，如 ollama / vllm / sglang。"""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """检查引擎依赖是否可用。"""
        pass

    @abc.abstractmethod
    def check_service(self, model_name: str) -> None:
        """检查服务/模型是否可用，不可用时抛出相应异常。"""
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
        """单轮推理：给定 prompt，返回模型生成文本。"""
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
        """多轮对话：OpenAI 风格 messages，返回助手回复。"""
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
        """结构化输出（默认返回原始文本，SGLang 可重写为 JSON）。"""
        return self.generate(
            prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


# ---------- Ollama 适配器 ----------
class OllamaAdapter(BaseLLMAdapter):
    """Ollama 引擎适配：基于 ollama 库或 httpx 调用本地 API，兼容多轮对话。"""

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
                r = c.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def check_service(self, model_name: str) -> None:
        if not OLLAMA_AVAILABLE:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(f"{self.base_url}/api/tags")
                    if r.status_code != 200:
                        raise EngineNotRunningError(
                            f"Ollama 服务未就绪: {self.base_url}，请先启动 Ollama。"
                        )
                    data = r.json()
                    names = [m.get("name") for m in data.get("models", [])]
                    if model_name not in names and not any(
                        m.startswith(model_name) or model_name in m for m in names
                    ):
                        raise ModelNotFoundError(
                            f"模型 '{model_name}' 不在 Ollama 已拉取列表中。可用: {names}。"
                        )
            except httpx.RequestError as e:
                raise EngineNotRunningError(
                    f"无法连接 Ollama 服务 {self.base_url}，请确认已启动。错误: {e}"
                )
            return
        # 使用 ollama 库
        try:
            ollama.list()
        except Exception as e:
            raise EngineNotRunningError(f"Ollama 服务未就绪: {e}")
        models = [m["name"] for m in ollama.list().get("models", [])]
        if model_name not in models and not any(
            m.startswith(model_name) or model_name in m for m in models
        ):
            raise ModelNotFoundError(f"模型 '{model_name}' 不存在。可用: {models}。")

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
        self.check_service(model_name)
        if OLLAMA_AVAILABLE:
            opts = {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                **{k: v for k, v in kwargs.items() if v is not None},
            }
            resp = ollama.generate(model=model_name, prompt=prompt, options=opts)
            return resp.get("response", "").strip()
        # httpx 回退
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
        }
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()

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
        self.check_service(model_name)
        if OLLAMA_AVAILABLE:
            opts = {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                **{k: v for k, v in kwargs.items() if v is not None},
            }
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                options=opts,
            )
            return resp.get("message", {}).get("content", "").strip()
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
        }
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()


# ---------- VLLM 适配器 ----------
class VLLMAdapter(BaseLLMAdapter):
    """VLLM 引擎适配：支持本地 LLM 实例或远程 OpenAI 兼容 API。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ):
        """
        base_url: 远程 API 地址（如 http://localhost:8000/v1），与 OpenAI 兼容。
        local_model_path: 本地模型路径，用于本地 LLM 实例；与 base_url 二选一。
        """
        self.base_url = base_url
        self.local_model_path = local_model_path
        self._llm: Any = None

    @property
    def engine_type(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        return VLLM_AVAILABLE

    def _get_llm(self, model_name: str) -> Any:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError(
                "未安装 vllm。请执行: pip install vllm（需 GPU 环境）。"
            )
        if self.base_url:
            return None
        path = self.local_model_path or model_name
        if self._llm is None:
            self._llm = LLM(model=path, trust_remote_code=True)
        return self._llm

    def check_service(self, model_name: str) -> None:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError(
                "未安装 vllm。请执行: pip install vllm（需 GPU 环境）。"
            )
        if self.base_url:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(f"{self.base_url.replace('/v1', '')}/health")
                    if r.status_code != 200:
                        raise EngineNotRunningError(
                            f"VLLM 服务未就绪: {self.base_url}。"
                        )
            except httpx.RequestError as e:
                raise EngineNotRunningError(
                    f"无法连接 VLLM 服务 {self.base_url}。错误: {e}"
                )
        else:
            self._get_llm(model_name)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将 OpenAI messages 转为单条 prompt 文本（简单拼接）。"""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant: ")
        return "\n".join(parts)

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
        self.check_service(model_name)
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if self.base_url:
            # 远程 OpenAI 兼容
            with httpx.Client(timeout=120.0) as c:
                r = c.post(
                    f"{self.base_url}/completions",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                )
                r.raise_for_status()
                choices = r.json().get("choices", [])
                return (choices[0].get("text", "") if choices else "").strip()
        llm = self._get_llm(model_name)
        outs = llm.generate([prompt], sampling)
        return (outs[0].outputs[0].text if outs and outs[0].outputs else "").strip()

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
        prompt = self._messages_to_prompt(messages)
        return self.generate(
            prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


# ---------- SGLang 适配器 ----------
class SGLangAdapter(BaseLLMAdapter):
    """SGLang 引擎适配：本地启动，支持 structured_generate 实现 JSON 结构化输出。"""

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
            raise EngineNotInstalledError(
                "未安装 sglang。请执行: pip install sglang（需 GPU 环境）。"
            )
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(f"{self.base_url}/get_model_info")
                if r.status_code != 200:
                    raise EngineNotRunningError(
                        f"SGLang 服务未就绪: {self.base_url}。"
                    )
        except httpx.RequestError as e:
            raise EngineNotRunningError(
                f"无法连接 SGLang 服务 {self.base_url}。错误: {e}"
            )

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
        self.check_service(model_name)
        # SGLang 常用 HTTP API：/generate 或兼容接口
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
            },
        }
        with httpx.Client(timeout=120.0) as c:
            r = c.post(f"{self.base_url}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("text", "") or data.get("generated_text", "") or "").strip()

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
        prompt = self._messages_to_prompt(messages)
        return self.generate(
            prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant: ")
        return "\n".join(parts)

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
        """SGLang 结构化输出：要求模型返回 JSON，解析后返回字典。"""
        self.check_service(model_name)
        # 在 prompt 中要求输出 JSON，便于解析
        schema_hint = ""
        if schema:
            schema_hint = f"\n请严格按以下 JSON 结构返回：\n{json.dumps(schema, ensure_ascii=False)}\n"
        full_prompt = f"{prompt}{schema_hint}\n请只输出合法 JSON，不要其他文字。"
        raw = self.generate(
            full_prompt,
            model_name=model_name,
            temperature=max(0.1, temperature),
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )
        try:
            # 尝试从回复中截取 JSON 块
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


# ---------- 核心推理类（对外唯一入口） ----------
class LLMInferencer:
    """主推理类：根据 engine_type 自动切换引擎，统一参数校验与引擎状态检测。"""

    ENGINE_MAP = {
        "ollama": OllamaAdapter,
        "vllm": VLLMAdapter,
        "sglang": SGLangAdapter,
    }

    def __init__(
        self,
        engine_type: Literal["ollama", "vllm", "sglang"],
        model_name: str = "llama3.2",
        *,
        ollama_base_url: str = "http://localhost:11434",
        vllm_base_url: Optional[str] = None,
        vllm_local_model_path: Optional[str] = None,
        sglang_base_url: str = "http://localhost:30000",
    ):
        """
        初始化推理引擎。
        engine_type: 引擎类型，ollama / vllm / sglang。
        model_name: 默认模型名，如 Llama3-8B、Qwen-7B、llama3.2。
        """
        if engine_type not in self.ENGINE_MAP:
            raise InvalidParameterError(
                f"engine_type 仅允许 ollama / vllm / sglang，当前: {engine_type}"
            )
        self.engine_type = engine_type
        self.model_name = model_name
        adapter_cls = self.ENGINE_MAP[engine_type]
        if engine_type == "ollama":
            self._adapter: BaseLLMAdapter = adapter_cls(base_url=ollama_base_url)
        elif engine_type == "vllm":
            self._adapter = adapter_cls(
                base_url=vllm_base_url,
                local_model_path=vllm_local_model_path,
            )
        else:
            self._adapter = adapter_cls(base_url=sglang_base_url)
        if not self._adapter.is_available():
            raise EngineNotInstalledError(
                f"引擎 '{engine_type}' 依赖未安装，请安装对应可选依赖。"
            )
        self._adapter.check_service(model_name)
        logger.info("LLMInferencer 初始化成功: engine=%s, model=%s", engine_type, model_name)

    def _validate_common(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> None:
        if not (0 <= temperature <= 2):
            raise InvalidParameterError("temperature 需在 [0, 2] 范围内")
        if max_tokens < 1:
            raise InvalidParameterError("max_tokens 需为正整数")
        if not (0 <= top_p <= 1):
            raise InvalidParameterError("top_p 需在 [0, 1] 范围内")

    def generate(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """单轮推理。"""
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.generate(
            prompt,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """多轮对话，OpenAI messages 格式。"""
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.chat(
            messages,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

    def structured_generate(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        *,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], str]:
        """结构化输出，仅 SGLang 实现 JSON 解析，其他引擎返回普通文本。"""
        if self.engine_type != "sglang":
            raise StructuredOutputNotSupportedError(
                "structured_generate 仅支持 SGLang 引擎。"
            )
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.structured_generate(
            prompt,
            schema,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


# ==================== FastAPI 接口层（与推理核心解耦） ====================

# 仅在使用 FastAPI 时导入，避免无 FastAPI 环境报错
try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    Request = None
    CORSMiddleware = None
    BaseModel = None
    Field = None
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:

    # ---------- Pydantic 请求/响应模型 ----------

    class GenerateRequest(BaseModel):
        """单轮推理请求体。"""
        prompt: str = Field(..., min_length=1, description="输入提示词")
        engine_type: Literal["ollama", "vllm", "sglang"] = Field(
            ..., description="引擎类型"
        )
        model_name: str = Field(default="llama3.2", min_length=1)
        temperature: float = Field(default=0.7, ge=0, le=2)
        max_tokens: int = Field(default=1024, gt=0)
        top_p: float = Field(default=0.95, ge=0, le=1)

    class ChatMessage(BaseModel):
        role: Literal["system", "user", "assistant"]
        content: str

    class ChatRequest(BaseModel):
        """多轮对话请求体。"""
        messages: List[ChatMessage] = Field(..., min_length=1)
        engine_type: Literal["ollama", "vllm", "sglang"] = Field(...)
        model_name: str = Field(default="llama3.2", min_length=1)
        temperature: float = Field(default=0.7, ge=0, le=2)
        max_tokens: int = Field(default=1024, gt=0)
        top_p: float = Field(default=0.95, ge=0, le=1)

    class StructuredGenerateRequest(BaseModel):
        """结构化输出请求体（仅 SGLang）。"""
        prompt: str = Field(..., min_length=1)
        engine_type: Literal["sglang"] = Field(...)
        model_name: str = Field(default="llama3.2", min_length=1)
        schema: Optional[Dict[str, Any]] = None
        temperature: float = Field(default=0.7, ge=0, le=2)
        max_tokens: int = Field(default=1024, gt=0)
        top_p: float = Field(default=0.95, ge=0, le=1)

    class SuccessData(BaseModel):
        """成功时的 data 体。"""
        engine_type: str
        model_name: str
        response: Union[str, Dict[str, Any]]
        cost_time: float

    class ApiResponse(BaseModel):
        """统一 API 响应格式。"""
        request_id: str
        code: int
        msg: str
        data: Optional[Union[SuccessData, Dict[str, Any]]] = None

    def _make_inferencer(engine_type: str, model_name: str) -> LLMInferencer:
        """根据 engine_type 和 model_name 创建 LLMInferencer（接口层不持有长期实例）。"""
        return LLMInferencer(engine_type=engine_type, model_name=model_name)

    def create_app() -> "FastAPI":
        """创建 FastAPI 应用：路由、全局异常、CORS、请求日志。"""
        app = FastAPI(
            title="LLM 推理统一 API",
            description="单轮/多轮/结构化输出，支持 Ollama/VLLM/SGLang",
            version="1.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.exception_handler(LLMInferenceError)
        def handle_llm_error(request: Request, exc: LLMInferenceError):
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            code = 400 if isinstance(
                exc,
                (InvalidParameterError, ModelNotFoundError, StructuredOutputNotSupportedError),
            ) else 500
            if isinstance(exc, EngineNotInstalledError):
                code = 500
            body = ApiResponse(
                request_id=request_id,
                code=code,
                msg=str(exc),
                data=None,
            )
            return JSONResponse(status_code=code, content=body.model_dump())

        @app.exception_handler(Exception)
        def handle_generic(request: Request, exc: Exception):
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.exception("未捕获异常: %s", exc)
            body = ApiResponse(
                request_id=request_id,
                code=500,
                msg=f"服务器内部错误: {exc!s}",
                data=None,
            )
            return JSONResponse(status_code=500, content=body.model_dump())

        @app.middleware("http")
        async def add_request_id_and_log(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            logger.info(
                "request_id=%s path=%s status=%s",
                request_id,
                request.url.path,
                response.status_code,
            )
            return response

        @app.post("/api/v1/llm/generate", response_model=ApiResponse)
        async def api_generate(body: GenerateRequest, request: Request):
            """单轮推理接口。"""
            rid = request.state.request_id
            logger.info(
                "request_id=%s engine=%s model=%s prompt_len=%d",
                rid,
                body.engine_type,
                body.model_name,
                len(body.prompt),
            )
            t0 = time.perf_counter()
            try:
                inferencer = _make_inferencer(body.engine_type, body.model_name)
                response = inferencer.generate(
                    body.prompt,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    top_p=body.top_p,
                )
                cost = time.perf_counter() - t0
                return ApiResponse(
                    request_id=rid,
                    code=200,
                    msg="success",
                    data=SuccessData(
                        engine_type=body.engine_type,
                        model_name=body.model_name,
                        response=response,
                        cost_time=round(cost, 4),
                    ),
                )
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("generate 异常: %s", e)
                raise

        @app.post("/api/v1/llm/chat", response_model=ApiResponse)
        async def api_chat(body: ChatRequest, request: Request):
            """多轮对话接口。"""
            rid = request.state.request_id
            messages = [{"role": m.role, "content": m.content} for m in body.messages]
            logger.info(
                "request_id=%s engine=%s model=%s messages=%d",
                rid,
                body.engine_type,
                body.model_name,
                len(messages),
            )
            t0 = time.perf_counter()
            try:
                inferencer = _make_inferencer(body.engine_type, body.model_name)
                response = inferencer.chat(
                    messages,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    top_p=body.top_p,
                )
                cost = time.perf_counter() - t0
                return ApiResponse(
                    request_id=rid,
                    code=200,
                    msg="success",
                    data=SuccessData(
                        engine_type=body.engine_type,
                        model_name=body.model_name,
                        response=response,
                        cost_time=round(cost, 4),
                    ),
                )
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("chat 异常: %s", e)
                raise

        @app.post("/api/v1/llm/structured-generate", response_model=ApiResponse)
        async def api_structured_generate(
            body: StructuredGenerateRequest, request: Request
        ):
            """结构化输出接口（仅 SGLang）。"""
            rid = request.state.request_id
            if body.engine_type != "sglang":
                raise InvalidParameterError(
                    "structured-generate 仅支持 engine_type=sglang"
                )
            logger.info(
                "request_id=%s engine=sglang model=%s",
                rid,
                body.model_name,
            )
            t0 = time.perf_counter()
            try:
                inferencer = _make_inferencer("sglang", body.model_name)
                result = inferencer.structured_generate(
                    body.prompt,
                    schema=body.schema,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    top_p=body.top_p,
                )
                cost = time.perf_counter() - t0
                return ApiResponse(
                    request_id=rid,
                    code=200,
                    msg="success",
                    data=SuccessData(
                        engine_type="sglang",
                        model_name=body.model_name,
                        response=result,
                        cost_time=round(cost, 4),
                    ),
                )
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("structured_generate 异常: %s", e)
                raise

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    # 供 uvicorn 或外部挂载使用
    app = create_app()

else:
    app = None


# ---------- 测试用例（原生 Python 调用） ----------
def run_core_tests() -> None:
    """运行核心推理测试：单轮、多轮、引擎检测；Ollama 可用时跑真实推理。"""
    logger.info("========== 开始核心推理模块测试 ==========")

    # 1) 引擎可用性
    logger.info("1. 引擎可用性检查")
    for eng in ("ollama", "vllm", "sglang"):
        try:
            inf = LLMInferencer(engine_type=eng, model_name="llama3.2")
            logger.info("  %s: 可用", eng)
        except EngineNotInstalledError as e:
            logger.info("  %s: 未安装 -> %s", eng, e)
        except (EngineNotRunningError, ModelNotFoundError) as e:
            logger.info("  %s: 服务/模型异常 -> %s", eng, e)

    # 2) 单轮推理（仅当 Ollama 可用且服务正常时）
    try:
        inf = LLMInferencer(engine_type="ollama", model_name="llama3.2")
        out = inf.generate("你好，请用一句话介绍你自己。", max_tokens=64)
        logger.info("2. 单轮推理(Ollama) 成功: %s", out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("2. 单轮推理 跳过或失败: %s", e)

    # 3) 多轮对话
    try:
        inf = LLMInferencer(engine_type="ollama", model_name="llama3.2")
        msgs = [
            {"role": "user", "content": "我叫小明。"},
            {"role": "assistant", "content": "你好小明！"},
            {"role": "user", "content": "你还记得我叫什么吗？"},
        ]
        out = inf.chat(msgs, max_tokens=64)
        logger.info("3. 多轮对话(Ollama) 成功: %s", out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("3. 多轮对话 跳过或失败: %s", e)

    # 4) 结构化输出（仅 SGLang）
    try:
        inf = LLMInferencer(engine_type="sglang", model_name="llama3.2")
        result = inf.structured_generate(
            "请输出一个包含 name 和 age 的 JSON。",
            schema={"name": "string", "age": "number"},
            max_tokens=128,
        )
        logger.info("4. 结构化输出(SGLang) 成功: %s", result)
    except StructuredOutputNotSupportedError:
        logger.info("4. 结构化输出 仅支持 SGLang，已跳过")
    except Exception as e:
        logger.info("4. 结构化输出 跳过或失败: %s", e)

    logger.info("========== 核心推理模块测试结束 ==========")


def run_api_tests(base_url: str = "http://127.0.0.1:8000") -> None:
    """调用 FastAPI 接口的简单测试（需先启动服务）。"""
    try:
        import requests
    except ImportError:
        logger.warning("未安装 requests，跳过 API 测试")
        return
    logger.info("========== API 接口测试（base_url=%s）==========", base_url)
    headers = {"Content-Type": "application/json"}

    # 单轮
    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/generate",
            json={
                "prompt": "你好",
                "engine_type": "ollama",
                "model_name": "llama3.2",
                "max_tokens": 32,
            },
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info("POST /api/v1/llm/generate -> code=%s data=%s", j.get("code"), j.get("data"))
    except Exception as e:
        logger.info("API generate 请求失败: %s", e)

    # 多轮
    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/chat",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "1+1=?"},
                ],
                "engine_type": "ollama",
                "model_name": "llama3.2",
                "max_tokens": 32,
            },
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info("POST /api/v1/llm/chat -> code=%s data=%s", j.get("code"), j.get("data"))
    except Exception as e:
        logger.info("API chat 请求失败: %s", e)

    # 结构化输出（仅 sglang）
    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/structured-generate",
            json={
                "prompt": "输出一个 JSON：{\"name\": \"张三\", \"age\": 20}",
                "engine_type": "sglang",
                "model_name": "llama3.2",
                "max_tokens": 64,
            },
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info(
            "POST /api/v1/llm/structured-generate -> code=%s data=%s",
            j.get("code"),
            j.get("data"),
        )
    except Exception as e:
        logger.info("API structured-generate 请求失败: %s", e)

    logger.info("========== API 接口测试结束 ==========")


# ---------- 使用示例（注释） ----------
"""
使用示例：

1) 初始化与单轮推理:
   inferencer = LLMInferencer(engine_type="ollama", model_name="llama3.2")
   text = inferencer.generate("介绍一下北京。", max_tokens=256)

2) 多轮推理:
   messages = [
       {"role": "user", "content": "我叫张三"},
       {"role": "assistant", "content": "你好张三！"},
       {"role": "user", "content": "我叫什么？"},
   ]
   reply = inferencer.chat(messages, max_tokens=128)

3) 引擎切换:
   inferencer_vllm = LLMInferencer(engine_type="vllm", model_name="Qwen/Qwen-7B")
   inferencer_sglang = LLMInferencer(engine_type="sglang", model_name="llama3.2")

4) 结构化输出（仅 SGLang）:
   result = inferencer_sglang.structured_generate(
       "请返回 {\"city\": \"北京\", \"population\": 数字}",
       schema={"city": "string", "population": "number"},
   )

常见问题：
- 服务未启动：先启动 Ollama/VLLM/SGLang 对应服务再调用。
- 显存不足：换小模型或使用 Ollama 等本地已优化方案。
- 库未安装：按 requirements.txt 安装对应引擎可选依赖。
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM 推理模块：测试或启动 API 服务")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="启动 FastAPI 服务",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API 服务 host")
    parser.add_argument("--port", type=int, default=8000, help="API 服务 port")
    parser.add_argument(
        "--api-test",
        action="store_true",
        help="在启动服务前先跑一次 API 测试（需本机已起服务）",
    )
    args = parser.parse_args()

    if args.serve:
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("请安装 fastapi 与 uvicorn: pip install fastapi uvicorn")
        import uvicorn
        logger.info("启动 API 服务: %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.api_test:
        run_api_tests(f"http://127.0.0.1:{args.port}")
    else:
        run_core_tests()
