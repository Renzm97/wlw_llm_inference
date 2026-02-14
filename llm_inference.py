#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动模型 API 服务：专注实现「启动模型」接口与运行实例管理。
推理能力与「验证启动的模型是否可用」由 inference_core 提供；模型从 HF 下载并存到配置文件中的路径，每个实例分配唯一 UID（run_id）。

运行方式:
  - 仅运行测试: python llm_inference.py
  - 启动 API 服务: python llm_inference.py --serve [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import httpx

# 推理与验证：从 inference_core 导入
from inference_core import (
    BUILTIN_MODELS,
    CONFIG,
    LLMInferencer,
    EngineNotInstalledError,
    EngineNotRunningError,
    InvalidParameterError,
    LLMInferenceError,
    ModelNotFoundError,
    StructuredOutputNotSupportedError,
    _ensure_model_downloaded,
    validate_model_usable,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_inference")

# ==================== 运行实例管理（供 FastAPI 使用） ====================
# 全局注册表：run_id -> { "inferencer", "model_id", "engine_type", "model_name", "created_at", "address" }
RUNNING_INSTANCES: Dict[str, Dict[str, Any]] = {}
_running_lock = threading.Lock()


def _parse_gpu_memory_util(gpu_count: Optional[str]) -> Optional[float]:
    """将前端 GPU 数量（auto 或数字）转为 vllm gpu_memory_utilization。"""
    if not gpu_count or str(gpu_count).strip().lower() == "auto":
        return None
    try:
        v = float(str(gpu_count).strip())
        return max(0.1, min(1.0, v)) if v <= 1 else None  # 若 >1 视为 GPU 卡数，暂不映射
    except ValueError:
        return None


def _ensure_ollama_running_and_model(
    base_url: str,
    ollama_model_name: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> None:
    """检查 Ollama 服务可达，若本地无该模型则拉取（ollama pull），并上报真实进度。"""
    def _report(p: int, msg: str) -> None:
        if progress_callback:
            progress_callback(p, msg)

    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{base_url.rstrip('/')}/api/tags")
            if r.status_code != 200:
                logger.warning("[Ollama] /api/tags 失败 base_url=%s status=%s", base_url, r.status_code)
                raise EngineNotRunningError(f"Ollama 服务未就绪: {base_url}")
            data = r.json()
            models = [m.get("name", "") or m.get("model", "") for m in data.get("models", [])]
            logger.info("[Ollama] /api/tags 成功 base_url=%s 已有模型: %s", base_url, models)
            if not any(
                name == ollama_model_name or name.startswith(ollama_model_name + ":") or ollama_model_name in name
                for name in models
            ):
                logger.info("Ollama 中未找到模型 %s，尝试拉取: ollama pull %s", ollama_model_name, ollama_model_name)
                _report(5, "正在拉取 Ollama 模型…")
                env = os.environ.copy()
                if "localhost" not in base_url:
                    host = base_url.replace("https://", "").replace("http://", "")
                    env["OLLAMA_HOST"] = host
                proc = subprocess.Popen(
                    ["ollama", "pull", ollama_model_name],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    env=env,
                    text=True,
                )
                # 解析 stderr 中的进度，如 "pulling manifest... 100%" 或 "pulling xyz... 45%"
                prog = re.compile(r"(\d+)\s*%")
                last_pct = 5
                if proc.stderr:
                    for line in iter(proc.stderr.readline, ""):
                        mo = prog.search(line)
                        if mo:
                            pct = min(95, max(last_pct, int(mo.group(1))))
                            last_pct = pct
                            _report(pct, f"拉取中… {pct}%")
                proc.wait(timeout=600)
                if proc.returncode != 0:
                    logger.warning("[Ollama] ollama pull 失败 model=%s returncode=%s", ollama_model_name, proc.returncode)
                    raise EngineNotRunningError(f"ollama pull 失败: 退出码 {proc.returncode}")
            _report(95, "Ollama 模型就绪")
            logger.info("[Ollama] 模型已就绪 base_url=%s model=%s", base_url, ollama_model_name)
    except httpx.RequestError as e:
        logger.warning("[Ollama] 连接失败 base_url=%s: %s", base_url, e)
        raise EngineNotRunningError(f"无法连接 Ollama 服务 {base_url}: {e}")


def _start_model_impl(
    model_id: str,
    engine_type: Literal["ollama", "vllm", "sglang"],
    *,
    api_base_url: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    format: Optional[str] = None,
    size: Optional[str] = None,
    quantization: Optional[str] = None,
    gpu_count: Optional[str] = None,
    replicas: int = 1,
    thought_mode: bool = False,
    parse_inference: bool = False,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    根据引擎类型启动模型，注册运行实例并返回 (run_id, 可访问地址)。
    若提供 progress_callback(percent, message)，将上报真实进度供前端展示。
    """
    def _report(p: int, msg: str) -> None:
        if progress_callback:
            progress_callback(p, msg)

    entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
    if not entry:
        raise ModelNotFoundError(f"未知内置模型: {model_id}")
    if engine_type not in (entry.get("engines") or []):
        raise InvalidParameterError(
            f"模型 {model_id} 不支持引擎 {engine_type}，支持: {entry.get('engines')}"
        )

    run_id = str(uuid.uuid4())
    address: str

    if engine_type == "ollama":
        _report(0, "正在检查 Ollama 服务…")
        ollama_cfg = CONFIG.get("ollama") or {}
        base_url = (ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        ollama_model_name = entry.get("ollama_name") or model_id
        logger.info("[Ollama 启动] run_id=%s base_url=%s model=%s 开始检查服务", run_id, base_url, ollama_model_name)
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{base_url.rstrip('/')}/api/tags")
            if r.status_code != 200:
                logger.warning("[Ollama 启动] 服务不可用 base_url=%s status=%s", base_url, r.status_code)
                raise EngineNotRunningError(f"Ollama 服务未就绪: {base_url}")
        address = base_url
        with _running_lock:
            RUNNING_INSTANCES[run_id] = {
                "inferencer": None,
                "model_id": model_id,
                "engine_type": engine_type,
                "model_name": ollama_model_name,
                "created_at": time.time(),
                "address": address,
                "format": format,
                "size": size,
                "quantization": quantization,
                "gpu_count": gpu_count,
                "replicas": replicas,
                "thought_mode": thought_mode,
                "parse_inference": parse_inference,
            }
        logger.info("[Ollama 启动] run_id=%s 已登记 RUNNING_INSTANCES，当前总数=%s", run_id, len(RUNNING_INSTANCES))
        try:
            logger.info("[Ollama 启动] run_id=%s 正在检查/拉取模型…", run_id)
            _ensure_ollama_running_and_model(base_url, ollama_model_name, progress_callback=progress_callback)
            _report(98, "正在验证模型…")
            logger.info("[Ollama 启动] run_id=%s 正在创建 LLMInferencer…", run_id)
            inferencer = LLMInferencer(
                engine_type="ollama",
                model_name=ollama_model_name,
                ollama_base_url=base_url,
            )
            logger.info("[Ollama 启动] run_id=%s LLMInferencer 创建成功，正在验证可用性", run_id)
            if not validate_model_usable(inferencer, max_tokens=5):
                logger.warning("Ollama 模型验证未通过，但已登记为运行实例 run_id=%s", run_id)
            with _running_lock:
                if run_id in RUNNING_INSTANCES:
                    RUNNING_INSTANCES[run_id]["inferencer"] = inferencer
            logger.info("[Ollama 启动] run_id=%s 已设置 inferencer，RUNNING_INSTANCES 保留", run_id)
        except Exception as e:
            logger.exception("[Ollama 启动] run_id=%s 启动失败: %s，已从 RUNNING_INSTANCES 移除", run_id, e)
            with _running_lock:
                if run_id in RUNNING_INSTANCES:
                    del RUNNING_INSTANCES[run_id]
                    logger.info("[Ollama 启动] run_id=%s 已删除，当前 RUNNING_INSTANCES 总数=%s", run_id, len(RUNNING_INSTANCES))
            raise
        _report(100, "就绪")
        logger.info("已启动 Ollama 模型 uid=%s model=%s address=%s", run_id, ollama_model_name, address)
        return run_id, address

    # vllm / sglang: 从 HF 下载并在进程内加载
    _report(5, "正在下载模型…")
    resolved = _ensure_model_downloaded(model_id)
    _report(50, "下载完成，正在加载…")
    vllm_path: Optional[str] = None
    vllm_gpu_util: Optional[float] = None
    if engine_type == "vllm":
        vllm_path = resolved
        vllm_gpu_util = _parse_gpu_memory_util(gpu_count)

    inferencer = LLMInferencer(
        engine_type=engine_type,
        model_name=model_id,
        vllm_local_model_path=vllm_path,
        vllm_gpu_memory_utilization=vllm_gpu_util,
    )
    _report(90, "正在验证模型…")
    if not validate_model_usable(inferencer, max_tokens=5):
        logger.warning("模型验证未通过，但已登记为运行实例 run_id=%s", run_id)
    _report(100, "就绪")

    # 返回可访问地址：优先使用传入的本 API 服务地址，便于前端展示
    address = (api_base_url or "").rstrip("/") if api_base_url else f"local:{run_id}"
    with _running_lock:
        RUNNING_INSTANCES[run_id] = {
            "inferencer": inferencer,
            "model_id": model_id,
            "engine_type": engine_type,
            "model_name": model_id,
            "created_at": time.time(),
            "address": address,
            "format": format,
            "size": size,
            "quantization": quantization,
            "gpu_count": gpu_count,
            "replicas": replicas,
            "thought_mode": thought_mode,
            "parse_inference": parse_inference,
        }
    logger.info("已启动模型 uid=%s model_id=%s engine=%s address=%s", run_id, model_id, engine_type, address)
    return run_id, address


def _user_facing_start_error(exc: Exception) -> str:
    """将启动模型时的异常转为用户可读的提示（如显存不足）。"""
    msg = str(exc).lower()
    if isinstance(exc, (ValueError, RuntimeError)) and (
        "memory" in msg or "gpu" in msg or "engine core initialization failed" in msg or "free memory" in msg
    ):
        return "GPU 显存不足，无法再加载新模型。请先停止「运行模型」中已有的 vLLM 实例，或在 config.json 中调低 vllm.gpu_memory_utilization 后重试。"
    return str(exc)


def _stop_model_impl(run_id: str) -> bool:
    """从注册表移除运行实例，返回是否曾存在。"""
    with _running_lock:
        existed = run_id in RUNNING_INSTANCES
        if existed:
            del RUNNING_INSTANCES[run_id]
        return existed


def _list_ollama_models_from_service() -> List[Dict[str, Any]]:
    """请求配置的 Ollama 服务 /api/tags，返回可用的模型列表（用于合并到 running 列表展示）。"""
    ollama_cfg = CONFIG.get("ollama") or {}
    base_url = (ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"{base_url}/api/tags")
            if r.status_code != 200:
                return []
            data = r.json()
            models = data.get("models") or []
    except Exception as e:
        logger.debug("获取 Ollama 模型列表失败 base_url=%s: %s", base_url, e)
        return []
    items = []
    for m in models:
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        model_id = name.split(":")[0] if ":" in name else name
        items.append({
            "run_id": f"ollama:{name}",
            "model_id": model_id,
            "engine_type": "ollama",
            "model_name": name,
            "address": base_url,
            "created_at": time.time(),
        })
    return items


def _get_running_inferencer(run_id: str) -> Optional[LLMInferencer]:
    """根据 run_id 获取已注册的 LLMInferencer，不存在返回 None；存在但 inferencer 未就绪则抛错。
    若 run_id 以 'ollama:' 开头，则按「Ollama 已发现模型」现场创建 Inferencer，不要求曾通过 API 启动。"""
    if run_id.startswith("ollama:"):
        model_name = run_id[7:].strip()
        if not model_name:
            return None
        ollama_cfg = CONFIG.get("ollama") or {}
        base_url = ollama_cfg.get("base_url") or "http://localhost:11434"
        inf = LLMInferencer(engine_type="ollama", model_name=model_name, ollama_base_url=base_url)
        return inf
    with _running_lock:
        entry = RUNNING_INSTANCES.get(run_id)
    if not entry:
        return None
    inf = entry.get("inferencer")
    if inf is None:
        raise EngineNotRunningError("该模型正在启动中（如拉取/加载），请稍候再试")
    return inf


# ==================== FastAPI 接口层（与推理核心解耦） ====================

# 仅在使用 FastAPI 时导入，避免无 FastAPI 环境报错
try:
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    Request = None
    CORSMiddleware = None
    JSONResponse = None
    FileResponse = None
    StaticFiles = None
    BaseModel = None
    Field = None
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:

    # ---------- Pydantic 请求/响应模型 ----------

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
        """启动模型请求体：配置引擎与参数后启动（ollama/vllm/sglang），分配唯一 UID，返回可访问地址。"""
        model_id: str = Field(..., min_length=1, description="内置模型 id，如 qwen2-0.5b、llama3.2")
        engine_type: Literal["ollama", "vllm", "sglang"] = Field(..., description="模型引擎（必选）")
        format: Optional[str] = Field(default="pytorch", description="模型格式（必选，如 pytorch/safetensors）")
        size: Optional[str] = Field(default="0.5B", description="模型大小（必选，如 0.5B/1B）")
        quantization: Optional[str] = Field(default="none", description="量化（必选，如 none/int4/int8）")
        gpu_count: Optional[str] = Field(default="auto", description="GPU 数量，auto 或数字")
        replicas: int = Field(default=1, ge=1, description="副本数")
        thought_mode: bool = Field(default=False, description="是否开启思考模式")
        parse_inference: bool = Field(default=False, description="是否解析推理内容")

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

        # ---------- 模型管理 API ----------
        @app.get("/api/v1/models", response_model=ApiResponse)
        async def api_list_models(request: Request):
            """列出内置模型元信息（id、名称、官方地址、量化、适配引擎等）。"""
            rid = request.state.request_id
            data = [
                {
                    "id": m["id"],
                    "name": m["name"],
                    "official_url": m.get("official_url"),
                    "quantizations": m.get("quantizations", []),
                    "engines": m.get("engines", []),
                }
                for m in BUILTIN_MODELS
            ]
            return ApiResponse(request_id=rid, code=200, msg="success", data={"models": data})

        @app.post("/api/v1/models/start", response_model=ApiResponse)
        async def api_start_model(body: StartModelRequest, request: Request):
            """根据前端命令启动 ollama/vllm/sglang 对应模型，返回 run_id 与可访问地址供运行模型页展示。"""
            rid = request.state.request_id
            api_base_url = str(request.base_url).rstrip("/")
            try:
                run_id, address = _start_model_impl(
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
                friendly = _user_facing_start_error(e)
                if friendly != str(e):
                    logger.warning("启动模型显存不足: %s", e)
                    raise EngineNotRunningError(friendly) from e
                raise
            except Exception as e:
                logger.exception("启动模型异常: %s", e)
                raise

        @app.post("/api/v1/models/start-stream")
        async def api_start_model_stream(body: StartModelRequest, request: Request):
            """流式启动模型：返回 NDJSON 流，每行 { progress, message } 或最终 { progress: 100, run_id, address }。"""
            api_base_url = str(request.base_url).rstrip("/")
            progress_queue: queue.Queue = queue.Queue()
            result_holder: List[Any] = []

            def progress_callback(percent: int, message: str) -> None:
                progress_queue.put({"progress": percent, "message": message})

            def run_start() -> None:
                try:
                    run_id, address = _start_model_impl(
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
                    result_holder.append(("err", _user_facing_start_error(e)))

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

        @app.get("/api/v1/models/running", response_model=ApiResponse)
        async def api_list_running(request: Request):
            """列出当前运行中的模型实例（仅包含通过本 API「启动」的实例，不自动发现 Ollama 已拉取模型）。"""
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

        @app.post("/api/v1/models/running/{run_id}/stop", response_model=ApiResponse)
        async def api_stop_model(run_id: str, request: Request):
            """停止并移除指定 run_id 的运行实例。"""
            rid = request.state.request_id
            existed = _stop_model_impl(run_id)
            return ApiResponse(
                request_id=rid,
                code=200,
                msg="success",
                data={"run_id": run_id, "stopped": existed},
            )

        def _resolve_inferencer_from_body(
            run_id: Optional[str],
            engine_type: Optional[str],
            model_name: str,
        ) -> LLMInferencer:
            """根据 run_id 或 engine_type+model_name 解析 inferencer。"""
            if run_id:
                inf = _get_running_inferencer(run_id)
                if inf is None:
                    raise InvalidParameterError(f"run_id 无效或已停止: {run_id}")
                return inf
            if not engine_type:
                raise InvalidParameterError("请提供 run_id 或 engine_type")
            return _make_inferencer(engine_type, model_name)

        @app.post("/api/v1/llm/generate", response_model=ApiResponse)
        async def api_generate(body: GenerateRequest, request: Request):
            """单轮推理接口。可传 run_id 使用已启动模型，或传 engine_type/model_name。"""
            rid = request.state.request_id
            inferencer = _resolve_inferencer_from_body(
                body.run_id, body.engine_type, body.model_name
            )
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
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("generate 异常: %s", e)
                raise

        @app.post("/api/v1/llm/chat", response_model=ApiResponse)
        async def api_chat(body: ChatRequest, request: Request):
            """多轮对话接口。可传 run_id 使用已启动模型，或传 engine_type/model_name。"""
            rid = request.state.request_id
            inferencer = _resolve_inferencer_from_body(
                body.run_id, body.engine_type, body.model_name
            )
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
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("chat 异常: %s", e)
                raise

        @app.post("/api/v1/llm/structured-generate", response_model=ApiResponse)
        async def api_structured_generate(
            body: StructuredGenerateRequest, request: Request
        ):
            """结构化输出接口（仅 SGLang）。可传 run_id 或 engine_type=sglang/model_name。"""
            rid = request.state.request_id
            inferencer = _resolve_inferencer_from_body(
                body.run_id, body.engine_type, body.model_name
            )
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
            except LLMInferenceError:
                raise
            except Exception as e:
                logger.exception("structured_generate 异常: %s", e)
                raise

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        # 前端静态资源：/ 返回 index.html，/css、/js 等走静态目录
        _frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        if os.path.isdir(_frontend_dir):
            app.mount("/css", StaticFiles(directory=os.path.join(_frontend_dir, "css")), name="css")
            app.mount("/js", StaticFiles(directory=os.path.join(_frontend_dir, "js")), name="js")

            @app.get("/")
            async def index():
                return FileResponse(os.path.join(_frontend_dir, "index.html"))

        return app

    # 供 uvicorn 或外部挂载使用
    app = create_app()

else:
    app = None


# ---------- 测试用例（原生 Python 调用） ----------
def run_core_tests() -> None:
    """运行核心推理测试：单轮、多轮、引擎检测。"""
    # 仅测试指定引擎：设为 "ollama" / "vllm" / "sglang" 只测该引擎；设为 None 测全部
    test_engine: Optional[str] = "vllm"

    logger.info("========== 开始核心推理模块测试 ==========")

    engines = ("ollama", "vllm", "sglang")
    if test_engine:
        engines = (test_engine,)
        logger.info("当前仅测试引擎: %s", test_engine)

    # 1) 引擎可用性（单引擎测试时复用此处创建的 inferencer，避免 vLLM 等重复加载占满显存）
    reusable_inf: Optional["LLMInferencer"] = None
    logger.info("1. 引擎可用性检查")
    for eng in engines:
        try:
            inf = LLMInferencer(engine_type=eng, model_name="llama3.2")
            if test_engine:
                reusable_inf = inf
            logger.info("  %s: 可用", eng)
        except EngineNotInstalledError as e:
            logger.info("  %s: 未安装 -> %s", eng, e)
        except (EngineNotRunningError, ModelNotFoundError, OSError) as e:
            logger.info("  %s: 服务/模型异常 -> %s", eng, e)

    # 2) 单轮推理
    run_eng = test_engine or "ollama"
    try:
        inf = reusable_inf if (reusable_inf and test_engine) else LLMInferencer(engine_type=run_eng, model_name="llama3.2")
        out = inf.generate("你好，请用一句话介绍你自己。", max_tokens=64)
        logger.info("2. 单轮推理(%s) 成功: %s", run_eng, out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("2. 单轮推理 跳过或失败: %s", e)

    # 3) 多轮对话
    try:
        inf = reusable_inf if (reusable_inf and test_engine) else LLMInferencer(engine_type=run_eng, model_name="llama3.2")
        msgs = [
            {"role": "user", "content": "我叫小明。"},
            {"role": "assistant", "content": "你好小明！"},
            {"role": "user", "content": "你还记得我叫什么吗？"},
        ]
        out = inf.chat(msgs, max_tokens=64)
        logger.info("3. 多轮对话(%s) 成功: %s", run_eng, out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("3. 多轮对话 跳过或失败: %s", e)

    # 4) 结构化输出（仅 SGLang）
    if not test_engine or test_engine == "sglang":
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
    else:
        logger.info("4. 结构化输出 已跳过（当前仅测试 %s）", run_eng)

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

    # 模型列表
    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=10)
        j = r.json()
        logger.info("GET /api/v1/models -> code=%s models_count=%s", j.get("code"), len((j.get("data") or {}).get("models") or []))
    except Exception as e:
        logger.info("GET /api/v1/models 请求失败: %s", e)

    # 运行中实例列表
    try:
        r = requests.get(f"{base_url}/api/v1/models/running", timeout=10)
        j = r.json()
        logger.info("GET /api/v1/models/running -> code=%s running_count=%s", j.get("code"), len((j.get("data") or {}).get("running") or []))
    except Exception as e:
        logger.info("GET /api/v1/models/running 请求失败: %s", e)

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
