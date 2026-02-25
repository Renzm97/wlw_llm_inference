# -*- coding: utf-8 -*-
"""
运行日志缓冲：记录启动/停止等关键事件及近期应用日志，供前端「运行日志」面板展示。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

# 内存缓冲，最多保留条数
MAX_LOGS = 500
_log_buffer: deque = deque(maxlen=MAX_LOGS)
_lock = threading.Lock()


def append_log(
    message: str,
    level: str = "INFO",
    *,
    run_id: Optional[str] = None,
    engine: Optional[str] = None,
    model_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> None:
    """追加一条运行日志（供 API 返回、前端展示）。"""
    if ts is None:
        ts = time.time()
    entry: Dict[str, Any] = {
        "ts": ts,
        "level": level.upper(),
        "message": message,
    }
    if run_id is not None:
        entry["run_id"] = run_id
    if engine is not None:
        entry["engine"] = engine
    if model_id is not None:
        entry["model_id"] = model_id
    with _lock:
        _log_buffer.append(entry)


def log_run_event(
    message: str,
    level: str = "INFO",
    *,
    run_id: Optional[str] = None,
    engine: Optional[str] = None,
    model_id: Optional[str] = None,
) -> None:
    """记录一次运行相关事件（启动成功、停止、失败等），便于区分是 Ollama 还是 vLLM 等。"""
    append_log(message, level=level, run_id=run_id, engine=engine, model_id=model_id)


def get_logs(limit: int = 200, level: Optional[str] = None) -> List[Dict[str, Any]]:
    """返回最近 limit 条日志；level 为 None 表示全部，否则只返回该级别及以上。"""
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    min_level = level_order.get((level or "").upper(), 0)
    with _lock:
        snapshot = list(_log_buffer)
    out: List[Dict[str, Any]] = []
    for i in range(len(snapshot) - 1, -1, -1):
        if len(out) >= limit:
            break
        e = snapshot[i]
        if level and level_order.get(e.get("level", "INFO"), 0) < min_level:
            continue
        out.append(dict(e))
    out.reverse()
    return out


class RuntimeLogHandler(logging.Handler):
    """将 Python logging 的 record 写入运行日志缓冲（仅 INFO 及以上）。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname or "INFO"
            msg = self.format(record)
            extra = getattr(record, "run_id", None), getattr(record, "engine", None), getattr(record, "model_id", None)
            append_log(msg, level=level, run_id=extra[0], engine=extra[1], model_id=extra[2], ts=record.created)
        except Exception:
            pass


def install_app_log_handler() -> None:
    """将 RuntimeLogHandler 挂到 services / api / core 的 logger，只收集本应用日志。"""
    handler = RuntimeLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    for name in ("services", "api", "core"):
        log = logging.getLogger(name)
        if any(isinstance(h, RuntimeLogHandler) for h in log.handlers):
            continue
        log.addHandler(handler)
    append_log("运行日志已就绪；启动/停止模型时会在此显示引擎类型（Ollama、vLLM、SGLang）及 run_id。", level="INFO")
