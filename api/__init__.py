# -*- coding: utf-8 -*-
"""API 层：FastAPI 应用与路由。"""

try:
    from api.app import create_app
    FASTAPI_AVAILABLE = True
except ImportError:
    create_app = None
    FASTAPI_AVAILABLE = False

__all__ = ["create_app", "FASTAPI_AVAILABLE"]
