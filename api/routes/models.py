# -*- coding: utf-8 -*-
"""模型管理 API：列表、启动、流式启动、运行中列表、停止。"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import queue
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

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


def _merge_quantization_repos(sizes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从所有 size 的 quantization_repos 中提取并集。"""
    merged: Dict[str, Any] = {}
    for s in sizes:
        for k, v in (s.get("quantization_repos") or {}).items():
            if k not in merged:
                merged[k] = v
    return merged


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
                "quantization_repos": _merge_quantization_repos(m.get("sizes", [])),
                "engines": m.get("engines", ["vllm", "ollama", "sglang"]),
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
                "sizes": [{"size": m.get("size", "1B"), "hf_repo": m.get("hf_repo"), "ms_repo": m.get("ms_repo"), "ollama_name": m.get("ollama_name")}],
                "formats": ["pytorch", "safetensors"],
            }
            for m in BUILTIN_MODELS
        ]
    return ApiResponse(request_id=rid, code=200, msg="success", data={"models": data})


@router.post("/start", response_model=ApiResponse)
async def api_start_model(body: StartModelRequest, request: Request):
    """根据前端命令启动 ollama/vllm/sglang 对应模型，返回 run_id 与可访问地址。
    同步阻塞的启动过程（模型下载、加载）会放入线程池执行，避免阻塞事件循环。"""
    rid = request.state.request_id
    try:
        loop = asyncio.get_event_loop()
        run_id, address = await loop.run_in_executor(
            None,
            functools.partial(
                start_model_impl,
                model_id=body.model_id,
                engine_type=body.engine_type,
                format=body.format,
                size=body.size,
                quantization=body.quantization,
                gpu_count=body.gpu_count,
                replicas=body.replicas,
                thought_mode=body.thought_mode,
                parse_inference=body.parse_inference,
            ),
        )
        return ApiResponse(
            request_id=rid,
            code=200,
            msg="success",
            data={"run_id": run_id, "address": address},
        )
    except (ModelNotFoundError, InvalidParameterError, EngineNotInstalledError, EngineNotRunningError):
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
    progress_queue: queue.Queue = queue.Queue()
    result_holder: List[Any] = []
    cancel_event = threading.Event()

    def progress_callback(percent: int, message: str) -> None:
        if cancel_event.is_set():
            raise RuntimeError("启动已被客户端取消")
        progress_queue.put({"progress": percent, "message": message})

    def run_start() -> None:
        try:
            run_id, address = start_model_impl(
                model_id=body.model_id,
                engine_type=body.engine_type,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                format=body.format,
                size=body.size,
                quantization=body.quantization,
                gpu_count=body.gpu_count,
                replicas=body.replicas,
                thought_mode=body.thought_mode,
                parse_inference=body.parse_inference,
            )
            result_holder.append(("ok", run_id, address))
        except RuntimeError as e:
            if "取消" in str(e):
                logger.info("启动任务因客户端断开而取消")
                result_holder.append(("err", "已取消"))
            else:
                logger.exception("启动模型流式异常: %s", e)
                result_holder.append(("err", user_facing_start_error(e)))
        except Exception as e:
            logger.exception("启动模型流式异常: %s", e)
            result_holder.append(("err", user_facing_start_error(e)))

    thread = threading.Thread(target=run_start)
    thread.start()

    async def ndjson_stream():
        start_ts = asyncio.get_event_loop().time()
        try:
            while True:
                # 客户端断开时主动终止
                if await request.is_disconnected():
                    logger.info("流式启动客户端已断开，终止后台任务")
                    cancel_event.set()
                    return
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
                # 整体超时保护（30 分钟），防止线程崩溃导致无限循环
                if asyncio.get_event_loop().time() - start_ts > 1800:
                    cancel_event.set()
                    yield json.dumps({"progress": 0, "error": "启动超时（超过30分钟）"}, ensure_ascii=False) + "\n"
                    return
                await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            logger.info("流式启动连接被取消")
            cancel_event.set()
            raise

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
