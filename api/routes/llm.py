# -*- coding: utf-8 -*-
"""LLM 推理 API：generate、chat、structured-generate。"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from typing import Dict, Optional

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

# 缓存无 run_id 时创建的 inferencer，避免 vLLM 本地模式每次请求都重复加载模型
# LRU 上限 10，防止内存/GPU 泄漏
_inferencer_cache: Dict[str, LLMInferencer] = {}
_inf_cache_lock = threading.Lock()
_MAX_INF_CACHE = 10


def _release_inferencer_gpu(inf: LLMInferencer) -> None:
    """尝试释放 inferencer 持有的 GPU 资源。"""
    try:
        adapter = getattr(inf, "_adapter", None)
        if adapter is not None:
            llm = getattr(adapter, "_llm", None)
            if llm is not None:
                import gc
                del adapter._llm
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    except Exception as e:
        logger.debug("清理缓存 inferencer 失败: %s", e)


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
    cache_key = f"{engine_type}:{model_name}"
    evicted: List[LLMInferencer] = []
    with _inf_cache_lock:
        cached = _inferencer_cache.get(cache_key)
        if cached is not None:
            return cached
        inf = LLMInferencer(engine_type=engine_type, model_name=model_name)
        _inferencer_cache[cache_key] = inf
        # 弹出最早插入的条目，在锁外释放 GPU
        while len(_inferencer_cache) > _MAX_INF_CACHE:
            key = next(iter(_inferencer_cache))
            old = _inferencer_cache.pop(key)
            evicted.append(old)
    for old in evicted:
        _release_inferencer_gpu(old)
    return inf


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
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                inferencer.generate,
                body.prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                top_p=body.top_p,
            ),
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
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                inferencer.chat,
                messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                top_p=body.top_p,
            ),
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(
                inferencer.structured_generate,
                body.prompt,
                schema=body.response_schema,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                top_p=body.top_p,
            ),
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
