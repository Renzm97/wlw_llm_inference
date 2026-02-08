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

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

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


def _start_model_impl(
    model_id: str,
    engine_type: Literal["vllm", "sglang"],
    *,
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
    从 HF 下载模型到配置文件中的路径，创建 LLMInferencer 并注册，分配唯一 UID（run_id）。
    可选校验：启动后调用 inference_core.validate_model_usable 验证模型是否可用。
    """
    entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
    if not entry:
        raise ModelNotFoundError(f"未知内置模型: {model_id}")
    if engine_type not in (entry.get("engines") or []):
        raise InvalidParameterError(
            f"模型 {model_id} 不支持引擎 {engine_type}，支持: {entry.get('engines')}"
        )
    # 下载到平台目录（配置文件 models_dir / models_subdir_hf）
    resolved = _ensure_model_downloaded(model_id)

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
    # 验证启动的模型是否可用（短文本生成）
    if not validate_model_usable(inferencer, max_tokens=5):
        logger.warning("模型验证未通过，但已登记为运行实例 run_id=%s", model_id)

    run_id = str(uuid.uuid4())  # 唯一 UID
    address = f"local:{run_id}"
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
    logger.info("已启动模型 uid=%s model_id=%s engine=%s", run_id, model_id, engine_type)
    return run_id, address


def _stop_model_impl(run_id: str) -> bool:
    """从注册表移除运行实例，返回是否曾存在。"""
    with _running_lock:
        existed = run_id in RUNNING_INSTANCES
        if existed:
            del RUNNING_INSTANCES[run_id]
        return existed


def _get_running_inferencer(run_id: str) -> Optional[LLMInferencer]:
    """根据 run_id 获取已注册的 LLMInferencer，不存在返回 None。"""
    with _running_lock:
        entry = RUNNING_INSTANCES.get(run_id)
    if not entry:
        return None
    return entry.get("inferencer")


# ==================== FastAPI 接口层（与推理核心解耦） ====================

# 仅在使用 FastAPI 时导入，避免无 FastAPI 环境报错
try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
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
        """启动模型请求体：配置引擎与参数后，从 HF 下载并启动，分配唯一 UID。"""
        model_id: str = Field(..., min_length=1, description="内置模型 id，如 qwen2-0.5b、llama3.2")
        engine_type: Literal["vllm", "sglang"] = Field(..., description="模型引擎（必选）")
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
            """从 HF 下载模型到配置路径并启动，分配唯一 UID（run_id），返回 run_id 与运行地址。"""
            rid = request.state.request_id
            try:
                run_id, address = _start_model_impl(
                    model_id=body.model_id,
                    engine_type=body.engine_type,
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
            except Exception as e:
                logger.exception("启动模型异常: %s", e)
                raise

        @app.get("/api/v1/models/running", response_model=ApiResponse)
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
                    }
                    for k, v in RUNNING_INSTANCES.items()
                ]
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
