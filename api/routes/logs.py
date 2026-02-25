# -*- coding: utf-8 -*-
"""运行日志 API：供前端「运行日志」面板拉取近期日志。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import ApiResponse
from services.runtime_log import get_logs

router = APIRouter(prefix="/api/v1", tags=["logs"])


@router.get("/logs")
async def api_get_logs(request: Request, limit: int = 200, level: str | None = None):
    """
    返回近期运行日志（启动/停止模型、引擎类型等），便于确认当前是 Ollama 还是 vLLM 等。
    limit: 最多返回条数，默认 200；level: 可选 DEBUG/INFO/WARNING/ERROR 过滤。
    """
    rid = getattr(request.state, "request_id", None)
    logs = get_logs(limit=limit, level=level)
    return ApiResponse(
        request_id=rid or "",
        code=200,
        msg="success",
        data={"logs": logs},
    )
