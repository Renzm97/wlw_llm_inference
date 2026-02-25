# -*- coding: utf-8 -*-
"""API 路由。"""

from api.routes.llm import router as llm_router
from api.routes.logs import router as logs_router
from api.routes.models import router as models_router

__all__ = ["models_router", "llm_router", "logs_router"]
