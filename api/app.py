# -*- coding: utf-8 -*-
"""FastAPI 应用创建：中间件、异常处理、路由挂载、静态资源。"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import llm_router, logs_router, models_router, test_proxy_router
from api.schemas import ApiResponse
from services.runtime_log import install_app_log_handler
from core.config import CONFIG
from core.exceptions import (
    EngineNotInstalledError,
    EngineNotRunningError,
    InvalidParameterError,
    LLMInferenceError,
    ModelNotFoundError,
    StructuredOutputNotSupportedError,
)

logger = logging.getLogger("api.app")


def create_app() -> FastAPI:
    """创建 FastAPI 应用：路由、全局异常、CORS、请求日志。"""
    install_app_log_handler()
    app = FastAPI(
        title="LLM 推理统一 API",
        description="单轮/多轮/结构化输出，支持 Ollama/VLLM/SGLang",
        version="1.0.0",
    )
    _cors_origins = CONFIG.get("cors_origins")
    if not _cors_origins:
        _cors_origins = ["*"]
    _allow_credentials = False if "*" in _cors_origins else bool(CONFIG.get("cors_allow_credentials", False))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_allow_credentials,
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
        if isinstance(exc, EngineNotRunningError):
            code = 503
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
            msg="服务器内部错误",
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

    app.include_router(models_router)
    app.include_router(llm_router)
    app.include_router(logs_router)
    app.include_router(test_proxy_router)

    @app.get("/health")
    async def health():
        import httpx
        from urllib.parse import urljoin
        checks: Dict[str, Any] = {"status": "ok"}
        # Ollama
        ollama_cfg = CONFIG.get("ollama") or {}
        ollama_url = (ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{ollama_url}/api/tags")
                checks["ollama"] = r.status_code == 200
        except Exception:
            checks["ollama"] = False
        # vLLM
        vllm_cfg = CONFIG.get("vllm") or {}
        vllm_url = vllm_cfg.get("base_url")
        if vllm_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(urljoin(vllm_url.rstrip("/") + "/", "health"))
                    checks["vllm"] = r.status_code == 200
            except Exception:
                checks["vllm"] = False
        else:
            checks["vllm"] = None  # 本地模式，不检查远程服务
        # SGLang
        sglang_cfg = CONFIG.get("sglang") or {}
        sglang_url = (sglang_cfg.get("base_url") or "http://localhost:30000").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{sglang_url}/get_model_info")
                checks["sglang"] = r.status_code == 200
        except Exception:
            checks["sglang"] = False
        # 若有任意已配置远程引擎不可用，返回 503
        configured_unavailable = [k for k, v in checks.items() if v is False]
        if configured_unavailable:
            checks["status"] = "degraded"
            return JSONResponse(status_code=503, content=checks)
        return checks

    # 前端静态资源
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _frontend_dir = os.path.join(_project_root, "frontend")
    if os.path.isdir(_frontend_dir):
        app.mount("/css", StaticFiles(directory=os.path.join(_frontend_dir, "css")), name="css")
        app.mount("/js", StaticFiles(directory=os.path.join(_frontend_dir, "js")), name="js")

        @app.get("/")
        async def index():
            return FileResponse(os.path.join(_frontend_dir, "index.html"))

        @app.get("/test")
        async def test_page():
            return FileResponse(os.path.join(_frontend_dir, "test.html"))

    return app
