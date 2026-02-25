# -*- coding: utf-8 -*-
"""LLM 推理 API：generate、chat、structured-generate。"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import Request

from api.schemas import (
    ApiResponse,
    ChatRequest,
    GenerateRequest,
    SuccessData,
    StructuredGenerateRequest,
)
from core import LLMInferencer
from core.exceptions import InvalidParameterError
from services import get_running_inferencer

logger = logging.getLogger("api.routes.llm")

# 使用与 models 相同的 router 前缀风格，由 app 挂载时统一加前缀
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


def _resolve_inferencer(
    run_id: Optional[str],
    engine_type: Optional[str],
    model_name: str,
) -> LLMInferencer:
    """根据 run_id 或 engine_type+model_name 解析 inferencer。"""
    if run_id:
        inf = get_running_inferencer(run_id)
        if inf is None:
            raise InvalidParameterError(f"run_id 无效或已停止: {run_id}")
        return inf
    if not engine_type:
        raise InvalidParameterError("请提供 run_id 或 engine_type")
    return LLMInferencer(engine_type=engine_type, model_name=model_name)


@router.post("/generate", response_model=ApiResponse)
async def api_generate(body: GenerateRequest, request: Request):
    """单轮推理接口。"""
    rid = request.state.request_id
    inferencer = _resolve_inferencer(body.run_id, body.engine_type, body.model_name)
    logger.info(
        "request_id=%s engine=%s model=%s prompt_len=%d",
        rid,
        inferencer.engine_type,
        inferencer.model_name,
        len(body.prompt),
    )
    t0 = time.perf_counter()
    try:
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
                engine_type=inferencer.engine_type,
                model_name=inferencer.model_name,
                response=response,
                cost_time=round(cost, 4),
            ),
        )
    except Exception as e:
        logger.exception("generate 异常: %s", e)
        raise


@router.post("/chat", response_model=ApiResponse)
async def api_chat(body: ChatRequest, request: Request):
    """多轮对话接口。"""
    rid = request.state.request_id
    inferencer = _resolve_inferencer(body.run_id, body.engine_type, body.model_name)
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    logger.info(
        "request_id=%s engine=%s model=%s messages=%d",
        rid,
        inferencer.engine_type,
        inferencer.model_name,
        len(messages),
    )
    t0 = time.perf_counter()
    try:
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
                engine_type=inferencer.engine_type,
                model_name=inferencer.model_name,
                response=response,
                cost_time=round(cost, 4),
            ),
        )
    except Exception as e:
        logger.exception("chat 异常: %s", e)
        raise


@router.post("/structured-generate", response_model=ApiResponse)
async def api_structured_generate(body: StructuredGenerateRequest, request: Request):
    """结构化输出接口（仅 SGLang）。"""
    rid = request.state.request_id
    inferencer = _resolve_inferencer(body.run_id, body.engine_type, body.model_name)
    if inferencer.engine_type != "sglang":
        raise InvalidParameterError(
            "structured-generate 仅支持 SGLang 引擎（run_id 对应引擎须为 sglang）"
        )
    logger.info(
        "request_id=%s engine=sglang model=%s",
        rid,
        inferencer.model_name,
    )
    t0 = time.perf_counter()
    try:
        result = inferencer.structured_generate(
            body.prompt,
            schema=body.response_schema,
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
                model_name=inferencer.model_name,
                response=result,
                cost_time=round(cost, 4),
            ),
        )
    except Exception as e:
        logger.exception("structured_generate 异常: %s", e)
        raise
