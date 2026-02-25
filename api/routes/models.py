# -*- coding: utf-8 -*-
"""模型管理 API：列表、启动、流式启动、运行中列表、停止。"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import uuid
from typing import Any, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.schemas import ApiResponse, StartModelRequest
from core import BUILTIN_MODELS, get_models_catalog
from core.exceptions import (
    EngineNotInstalledError,
    EngineNotRunningError,
    InvalidParameterError,
    ModelNotFoundError,
)
from services import (
    RUNNING_INSTANCES,
    _running_lock,
    start_model_impl,
    stop_model_impl,
    user_facing_start_error,
)

logger = logging.getLogger("api.routes.models")

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ApiResponse)
async def api_list_models(request: Request):
    """
    列出模型目录，供前端展示模型卡片与参数配置。
    若存在 models.json 则返回完整目录（含 sizes、quantizations、engines、formats）；
    否则返回扁平列表兼容旧版。
    """
    rid = request.state.request_id
    catalog = get_models_catalog()
    if catalog:
        data = [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "description": m.get("description", ""),
                "official_url": m.get("official_url"),
                "sizes": m.get("sizes", []),
                "quantizations": m.get("quantizations", ["none"]),
                "engines": m.get("engines", ["ollama", "vllm", "sglang"]),
                "formats": m.get("formats", ["pytorch", "safetensors"]),
            }
            for m in catalog
        ]
    else:
        data = [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "official_url": m.get("official_url"),
                "quantizations": m.get("quantizations", []),
                "engines": m.get("engines", []),
                "sizes": [{"size": m.get("size", "1B"), "hf_repo": m.get("hf_repo"), "ollama_name": m.get("ollama_name")}],
                "formats": ["pytorch", "safetensors"],
            }
            for m in BUILTIN_MODELS
        ]
    return ApiResponse(request_id=rid, code=200, msg="success", data={"models": data})


@router.post("/start", response_model=ApiResponse)
async def api_start_model(body: StartModelRequest, request: Request):
    """根据前端命令启动 ollama/vllm/sglang 对应模型，返回 run_id 与可访问地址。"""
    rid = request.state.request_id
    api_base_url = str(request.base_url).rstrip("/")
    try:
        run_id, address = start_model_impl(
            model_id=body.model_id,
            engine_type=body.engine_type,
            api_base_url=api_base_url,
            format=body.format,
            size=body.size,
            quantization=body.quantization,
            gpu_count=body.gpu_count,
            replicas=body.replicas,
            thought_mode=body.thought_mode,
            parse_inference=body.parse_inference,
        )
        return ApiResponse(
            request_id=rid,
            code=200,
            msg="success",
            data={"uid": run_id, "run_id": run_id, "address": address},
        )
    except (ModelNotFoundError, InvalidParameterError, EngineNotInstalledError):
        raise
    except (ValueError, RuntimeError) as e:
        friendly = user_facing_start_error(e)
        if friendly != str(e):
            logger.warning("启动模型显存不足: %s", e)
            raise EngineNotRunningError(friendly) from e
        raise
    except Exception as e:
        logger.exception("启动模型异常: %s", e)
        raise


@router.post("/start-stream")
async def api_start_model_stream(body: StartModelRequest, request: Request):
    """流式启动模型：返回 NDJSON 流。"""
    api_base_url = str(request.base_url).rstrip("/")
    progress_queue: queue.Queue = queue.Queue()
    result_holder: List[Any] = []

    def progress_callback(percent: int, message: str) -> None:
        progress_queue.put({"progress": percent, "message": message})

    def run_start() -> None:
        try:
            run_id, address = start_model_impl(
                model_id=body.model_id,
                engine_type=body.engine_type,
                api_base_url=api_base_url,
                progress_callback=progress_callback,
                format=body.format,
                size=body.size,
                quantization=body.quantization,
                gpu_count=body.gpu_count,
                replicas=body.replicas,
                thought_mode=body.thought_mode,
                parse_inference=body.parse_inference,
            )
            result_holder.append(("ok", run_id, address))
        except Exception as e:
            logger.exception("启动模型流式异常: %s", e)
            result_holder.append(("err", user_facing_start_error(e)))

    thread = threading.Thread(target=run_start)
    thread.start()

    async def ndjson_stream():
        while True:
            while not progress_queue.empty():
                try:
                    item = progress_queue.get_nowait()
                    yield json.dumps(item, ensure_ascii=False) + "\n"
                except queue.Empty:
                    break
            if result_holder:
                status = result_holder[0]
                if status[0] == "ok":
                    yield json.dumps(
                        {"progress": 100, "run_id": status[1], "address": status[2]},
                        ensure_ascii=False,
                    ) + "\n"
                else:
                    yield json.dumps({"progress": 0, "error": status[1]}, ensure_ascii=False) + "\n"
                return
            await asyncio.sleep(0.12)

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/running", response_model=ApiResponse)
async def api_list_running(request: Request):
    """列出当前运行中的模型实例。"""
    rid = request.state.request_id
    with _running_lock:
        items = [
            {
                "run_id": k,
                "model_id": v["model_id"],
                "engine_type": v["engine_type"],
                "model_name": v["model_name"],
                "address": v["address"],
                "created_at": v["created_at"],
                "gpu_count": v.get("gpu_count"),
                "quantization": v.get("quantization"),
                "size": v.get("size"),
                "replicas": v.get("replicas"),
            }
            for k, v in RUNNING_INSTANCES.items()
        ]
    logger.debug("GET /api/v1/models/running 返回 %s 条", len(items))
    return ApiResponse(request_id=rid, code=200, msg="success", data={"running": items})


@router.post("/running/{run_id}/stop", response_model=ApiResponse)
async def api_stop_model(run_id: str, request: Request):
    """停止并移除指定 run_id 的运行实例。"""
    rid = request.state.request_id
    existed = stop_model_impl(run_id)
    return ApiResponse(
        request_id=rid,
        code=200,
        msg="success",
        data={"run_id": run_id, "stopped": existed},
    )
