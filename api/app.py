# -*- coding: utf-8 -*-
"""FastAPI 应用创建：中间件、异常处理、路由挂载、静态资源。"""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import llm_router, logs_router, models_router
from api.schemas import ApiResponse
from services.runtime_log import install_app_log_handler
from core.exceptions import (
    EngineNotInstalledError,
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

    app.include_router(models_router)
    app.include_router(llm_router)
    app.include_router(logs_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # 前端静态资源
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _frontend_dir = os.path.join(_project_root, "frontend")
    if os.path.isdir(_frontend_dir):
        app.mount("/css", StaticFiles(directory=os.path.join(_frontend_dir, "css")), name="css")
        app.mount("/js", StaticFiles(directory=os.path.join(_frontend_dir, "js")), name="js")

        @app.get("/")
        async def index():
            return FileResponse(os.path.join(_frontend_dir, "index.html"))

    return app
