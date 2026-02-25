# -*- coding: utf-8 -*-
"""API 请求/响应 Pydantic 模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """单轮推理请求体。run_id 与 (engine_type, model_name) 二选一。"""
    prompt: str = Field(..., min_length=1, description="输入提示词")
    run_id: Optional[str] = Field(default=None, description="已启动模型的 run_id，与 engine_type/model_name 二选一")
    engine_type: Optional[Literal["ollama", "vllm", "sglang"]] = Field(default=None, description="引擎类型")
    model_name: str = Field(default="llama3.2", min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=0.95, ge=0, le=1)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """多轮对话请求体。run_id 与 (engine_type, model_name) 二选一。"""
    messages: List[ChatMessage] = Field(..., min_length=1)
    run_id: Optional[str] = Field(default=None, description="已启动模型的 run_id")
    engine_type: Optional[Literal["ollama", "vllm", "sglang"]] = Field(default=None)
    model_name: str = Field(default="llama3.2", min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=0.95, ge=0, le=1)


class StructuredGenerateRequest(BaseModel):
    """结构化输出请求体（仅 SGLang）。run_id 与 (engine_type, model_name) 二选一。"""
    prompt: str = Field(..., min_length=1)
    run_id: Optional[str] = Field(default=None, description="已启动模型的 run_id")
    engine_type: Optional[Literal["sglang"]] = Field(default=None)
    model_name: str = Field(default="llama3.2", min_length=1)
    response_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")
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


class StartModelRequest(BaseModel):
    """启动模型请求体。"""
    model_id: str = Field(..., min_length=1, description="内置模型 id，如 qwen2-0.5b、llama3.2")
    engine_type: Literal["ollama", "vllm", "sglang"] = Field(..., description="模型引擎（必选）")
    format: Optional[str] = Field(default="pytorch", description="模型格式（必选，如 pytorch/safetensors）")
    size: Optional[str] = Field(default="0.5B", description="模型大小（必选，如 0.5B/1B）")
    quantization: Optional[str] = Field(default="none", description="量化（必选，如 none/int4/int8）")
    gpu_count: Optional[str] = Field(default="auto", description="GPU 数量，auto 或数字")
    replicas: int = Field(default=1, ge=1, description="副本数")
    thought_mode: bool = Field(default=False, description="是否开启思考模式")
    parse_inference: bool = Field(default=False, description="是否解析推理内容")
