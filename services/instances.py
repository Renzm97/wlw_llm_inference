# -*- coding: utf-8 -*-
"""
运行实例管理：启动/停止模型、注册表、根据 run_id 获取 Inferencer。
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

import httpx

from core import (
    BUILTIN_MODELS,
    CONFIG,
    LLMInferencer,
    ensure_model_downloaded,
    get_model_variant,
    validate_model_usable,
)
from core.exceptions import (
    EngineNotRunningError,
    InvalidParameterError,
    ModelNotFoundError,
)
from services import runtime_log

logger = logging.getLogger("services.instances")

# 全局注册表：run_id -> { "inferencer", "model_id", "engine_type", "model_name", "created_at", "address", ... }
RUNNING_INSTANCES: Dict[str, Dict[str, Any]] = {}
_running_lock = threading.Lock()
_ollama_startup_lock = threading.Lock()
_ollama_serve_pid: Optional[int] = None


def _cleanup_ollama_serve() -> None:
    """程序退出时尝试终止自动启动的 ollama serve 进程。"""
    global _ollama_serve_pid
    pid = _ollama_serve_pid
    if pid is None:
        return
    try:
        import os, signal
        os.kill(pid, signal.SIGTERM)
        logger.info("已发送 SIGTERM 到 ollama serve 进程 pid=%s", pid)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning("清理 ollama serve 进程失败 pid=%s: %s", pid, e)


import atexit
atexit.register(_cleanup_ollama_serve)


def _parse_gpu_memory_util(gpu_count: Optional[str]) -> Optional[float]:
    """将前端 GPU 数量/利用率（auto 或 0~1 小数）转为 vllm gpu_memory_utilization。"""
    if not gpu_count or str(gpu_count).strip().lower() == "auto":
        return None
    try:
        v = float(str(gpu_count).strip())
        if v <= 0 or v > 1:
            logger.warning("gpu_count/gpu_util 值 %s 不在 (0,1] 范围内，已忽略", v)
            return None
        return max(0.1, min(1.0, v))
    except ValueError:
        logger.warning("gpu_count/gpu_util 值 %s 无法解析为数字，已忽略", gpu_count)
        return None


def _is_ollama_local(base_url: str) -> bool:
    """判断配置的 base_url 是否为本地（仅本地时才尝试自动启动 ollama serve）。"""
    u = (base_url or "").strip().rstrip("/").lower()
    if not u:
        return True
    return "localhost" in u or u.startswith("http://127.0.0.1") or u.startswith("http://127.0.0.1:")


def _check_ollama_reachable(base_url: str, timeout: float = 5.0) -> bool:
    """检测 Ollama 服务是否可达（GET /api/tags 返回 200）。"""
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _ensure_ollama_process_running(base_url: str, wait_seconds: int = 35) -> None:
    """
    若 Ollama 未运行且 base_url 为本地，则自动启动 ollama serve，并轮询直到就绪。
    非本地地址不尝试启动，直接由后续检查抛出 EngineNotRunningError。
    使用锁保护「检测-启动」临界区，防止并发请求启动多个 ollama serve 进程。
    """
    if _check_ollama_reachable(base_url, timeout=3.0):
        return
    if not _is_ollama_local(base_url):
        raise EngineNotRunningError(
            f"Ollama 服务未就绪: {base_url}（远程地址不会自动启动，请先在目标机器上运行 ollama serve）"
        )
    with _ollama_startup_lock:
        # 再次检查，可能其他线程已启动成功
        if _check_ollama_reachable(base_url, timeout=3.0):
            return
        logger.info("[Ollama] 本地服务未响应，尝试自动启动 ollama serve…")
        try:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
                start_new_session=True,
            )
            global _ollama_serve_pid
            _ollama_serve_pid = proc.pid
        except FileNotFoundError:
            raise EngineNotRunningError(
                "未找到 ollama 命令，请先安装 Ollama（https://ollama.com）并确保 ollama 在 PATH 中"
            )
        except Exception as e:
            raise EngineNotRunningError(f"启动 ollama serve 失败: {e}")
        url = base_url.rstrip("/")
        for _ in range(wait_seconds):
            time.sleep(1)
            if proc.poll() is not None:
                break
            if _check_ollama_reachable(url, timeout=2.0):
                logger.info("[Ollama] ollama serve 已就绪")
                runtime_log.log_run_event("[Ollama] 已自动启动本地 ollama serve", level="INFO", engine="ollama")
                return
        try:
            proc.terminate()
        except Exception:
            pass
        raise EngineNotRunningError(
            f"已启动 ollama serve 进程但 {wait_seconds} 秒内未就绪，请检查端口是否被占用或手动运行 ollama serve"
        )


def _ensure_ollama_running_and_model(
    base_url: str,
    ollama_model_name: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> None:
    """检查 Ollama 服务可达（未运行则本地自动启动），若本地无该模型则用 ollama pull 拉取，并上报真实进度。"""
    def _report(p: int, msg: str) -> None:
        if progress_callback:
            progress_callback(p, msg)

    _ensure_ollama_process_running(base_url)
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
                name == ollama_model_name or name.startswith(ollama_model_name + ":")
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
                prog = re.compile(r"(\d+)\s*%")
                last_pct = 5
                if proc.stderr:
                    # 使用 select 实现带超时的逐行读取，避免 stderr 无输出时永久阻塞
                    import select
                    start_ts = time.time()
                    while True:
                        remaining = 600 - (time.time() - start_ts)
                        if remaining <= 0:
                            break
                        ready, _, _ = select.select([proc.stderr], [], [], 1.0)
                        if ready:
                            line = proc.stderr.readline()
                            if not line:
                                break
                            mo = prog.search(line)
                            if mo:
                                pct = min(95, max(last_pct, int(mo.group(1))))
                                last_pct = pct
                                _report(pct, f"拉取中… {pct}%")
                        if proc.poll() is not None:
                            break
                proc.wait(timeout=600)
                if proc.returncode != 0:
                    logger.warning("[Ollama] ollama pull 失败 model=%s returncode=%s", ollama_model_name, proc.returncode)
                    raise EngineNotRunningError(f"ollama pull 失败: 退出码 {proc.returncode}")
            _report(95, "Ollama 模型就绪")
            logger.info("[Ollama] 模型已就绪 base_url=%s model=%s", base_url, ollama_model_name)
    except httpx.RequestError as e:
        logger.warning("[Ollama] 连接失败 base_url=%s: %s", base_url, e)
        raise EngineNotRunningError(f"无法连接 Ollama 服务 {base_url}: {e}")


def start_model_impl(
    model_id: str,
    engine_type: Literal["ollama", "vllm", "sglang"],
    *,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    format: Optional[str] = None,
    size: Optional[str] = None,
    quantization: Optional[str] = None,
    gpu_count: Optional[str] = None,
    replicas: int = 1,
    thought_mode: bool = False,
    parse_inference: bool = False,
    cancel_event: Optional[threading.Event] = None,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    根据引擎类型启动模型，注册运行实例并返回 (run_id, 可访问地址)。
    若提供 progress_callback(percent, message)，将上报真实进度供前端展示。
    若提供 cancel_event，将在关键节点检查取消信号。
    """
    def _report(p: int, msg: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("启动已被客户端取消")
        if progress_callback:
            progress_callback(p, msg)

    # 优先从 models.json 解析 variant（model_id + size）
    variant = get_model_variant(model_id, size) if size else None
    if variant:
        entry = variant
        ollama_name = entry.get("ollama_name") or model_id
        engines = entry.get("engines") or []
    else:
        # 未传 size 时尝试从 MODELS_CATALOG 查找第一个 size
        from core.config import MODELS_CATALOG
        catalog_entry = next((m for m in MODELS_CATALOG if m.get("id") == model_id), None)
        if catalog_entry and catalog_entry.get("sizes"):
            first = catalog_entry["sizes"][0]
            entry = {
                "ollama_name": first.get("ollama_name") or model_id,
                "engines": catalog_entry.get("engines") or ["vllm", "ollama", "sglang"],
            }
            ollama_name = entry["ollama_name"]
            engines = entry["engines"]
        else:
            # 兼容旧逻辑：从 BUILTIN_MODELS 查找
            entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
            if not entry:
                raise ModelNotFoundError(f"未知模型: {model_id}" + (f"（请指定 size，如 {size}）" if size else ""))
            ollama_name = entry.get("ollama_name") or model_id
            engines = entry.get("engines") or []

    if engine_type not in engines:
        raise InvalidParameterError(
            f"模型 {model_id} 不支持引擎 {engine_type}，支持: {engines}"
        )

    run_id = str(uuid.uuid4())
    address: str

    if engine_type == "ollama":
        _report(0, "正在检查 Ollama 服务…")
        ollama_cfg = CONFIG.get("ollama") or {}
        base_url = (ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        ollama_model_name = ollama_name
        logger.info("[Ollama 启动] run_id=%s base_url=%s model=%s 开始检查服务", run_id, base_url, ollama_model_name)
        _ensure_ollama_process_running(base_url)
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
                raise EngineNotRunningError(f"Ollama 模型 {ollama_model_name} 验证未通过，可能无法正常推理")
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
        runtime_log.log_run_event(
            f"[Ollama] 已启动模型 run_id={run_id} model={ollama_model_name} address={address}",
            level="INFO",
            run_id=run_id,
            engine="ollama",
            model_id=model_id,
        )
        return run_id, address

    # vllm: 从平台模型源下载并在进程内加载；sglang: 仅支持远程 API，不下载
    if engine_type == "sglang":
        _report(5, "正在连接 SGLang 服务…")
        inferencer = LLMInferencer(
            engine_type=engine_type,
            model_name=model_id,
        )
        _report(90, "正在验证模型…")
        if not validate_model_usable(inferencer, max_tokens=5):
            raise EngineNotRunningError(f"SGLang 模型 {model_id} 验证未通过，请确认 SGLang 服务已启动且模型已加载")
        _report(100, "就绪")
        sglang_cfg = CONFIG.get("sglang") or {}
        sglang_url = sglang_cfg.get("base_url", "http://localhost:30000")
        address = sglang_url.rstrip("/")
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
        runtime_log.log_run_event(
            f"[{engine_type.upper()}] 已启动模型 run_id={run_id} model_id={model_id} address={address}",
            level="INFO",
            run_id=run_id,
            engine=engine_type,
            model_id=model_id,
        )
        return run_id, address

    # vllm: 下载 + 本地加载
    _report(5, "正在下载模型…")
    resolved = ensure_model_downloaded(model_id, size=size, quantization=quantization)
    _report(50, "下载完成，正在加载…")
    vllm_path: Optional[str] = resolved
    vllm_gpu_util = _parse_gpu_memory_util(gpu_count)

    # 解析 vLLM 量化算法名：从 variant（size 级别）的 quantization_repos 中获取
    vllm_quant: Optional[str] = None
    if quantization and quantization != "none":
        q_repos = (variant or {}).get("quantization_repos") or {}
        if q_repos and quantization in q_repos:
            vllm_quant = q_repos[quantization].get("quantization")

    inferencer = LLMInferencer(
        engine_type=engine_type,
        model_name=vllm_path,
        vllm_local_model_path=vllm_path,
        vllm_gpu_memory_utilization=vllm_gpu_util,
        vllm_quantization=vllm_quant,
    )
    _report(90, "正在验证模型…")
    if not validate_model_usable(inferencer, max_tokens=5):
        raise EngineNotRunningError(f"vLLM 模型 {model_id} 验证未通过，可能因显存不足或模型文件损坏")
    _report(100, "就绪")

    # 地址应指向实际引擎服务地址，便于前端跳转或外部访问
    if engine_type == "vllm":
        vllm_cfg = CONFIG.get("vllm") or {}
        vllm_url = vllm_cfg.get("base_url")
        address = vllm_url.rstrip("/") if vllm_url else f"local:{run_id}"
    else:
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
    logger.info("已启动模型 uid=%s model_id=%s engine=%s address=%s", run_id, model_id, engine_type, address)
    runtime_log.log_run_event(
        f"[{engine_type.upper()}] 已启动模型 run_id={run_id} model_id={model_id} address={address}",
        level="INFO",
        run_id=run_id,
        engine=engine_type,
        model_id=model_id,
    )
    return run_id, address


def user_facing_start_error(exc: Exception) -> str:
    """将启动模型时的异常转为用户可读的提示（如显存不足）。"""
    msg = str(exc).lower()
    if isinstance(exc, (ValueError, RuntimeError)) and (
        "memory" in msg or "gpu" in msg or "engine core initialization failed" in msg or "free memory" in msg
    ):
        return "GPU 显存不足，无法再加载新模型。请先停止「运行模型」中已有的 vLLM 实例，或在 config.json 中调低 vllm.gpu_memory_utilization 后重试。"
    return str(exc)


def stop_model_impl(run_id: str) -> bool:
    """从注册表移除运行实例，尝试释放 GPU/进程资源，返回是否曾存在。"""
    with _running_lock:
        existed = run_id in RUNNING_INSTANCES
        entry = RUNNING_INSTANCES.pop(run_id, None)
        engine = entry.get("engine_type") if entry else None
        model_id = entry.get("model_id") if entry else None
        inferencer = entry.get("inferencer") if entry else None
    if existed:
        # 尝试释放 vLLM 本地实例的显存
        if engine == "vllm" and inferencer is not None:
            try:
                adapter = getattr(inferencer, "_adapter", None)
                if adapter is not None:
                    llm = getattr(adapter, "_llm", None)
                    if llm is not None:
                        import gc
                        del adapter._llm
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        logger.info("已释放 vLLM 实例显存 run_id=%s", run_id)
            except Exception as e:
                logger.warning("释放 vLLM 实例显存失败 run_id=%s: %s", run_id, e)
        runtime_log.log_run_event(
            f"已停止模型 run_id={run_id} engine={engine or '?'} model_id={model_id or '?'}",
            level="INFO",
            run_id=run_id,
            engine=engine,
            model_id=model_id,
        )
    return existed


def list_ollama_models_from_service() -> List[Dict[str, Any]]:
    """请求配置的 Ollama 服务 /api/tags，返回可用的模型列表。"""
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


# ollama: 前缀 inferencer 缓存，避免高频请求时重复创建
from collections import OrderedDict
_MAX_OLLAMA_CACHE = 50
_ollama_inferencer_cache: "OrderedDict[str, LLMInferencer]" = OrderedDict()
_ollama_inf_lock = threading.Lock()


def get_running_inferencer(run_id: str) -> Optional[LLMInferencer]:
    """
    根据 run_id 获取已注册的 LLMInferencer，不存在返回 None；
    存在但 inferencer 未就绪则抛 EngineNotRunningError。
    若 run_id 以 'ollama:' 开头，则按「Ollama 已发现模型」现场创建 Inferencer（带缓存）。
    """
    if run_id.startswith("ollama:"):
        model_name = run_id[7:].strip()
        if not model_name:
            return None
        with _ollama_inf_lock:
            cached = _ollama_inferencer_cache.get(run_id)
            if cached is not None:
                # LRU：移到末尾表示最近使用
                _ollama_inferencer_cache.move_to_end(run_id)
                return cached
            ollama_cfg = CONFIG.get("ollama") or {}
            base_url = ollama_cfg.get("base_url") or "http://localhost:11434"
            try:
                inf = LLMInferencer(engine_type="ollama", model_name=model_name, ollama_base_url=base_url)
            except Exception as e:
                logger.warning("现场创建 Ollama inferencer 失败 model=%s: %s", model_name, e)
                return None
            _ollama_inferencer_cache[run_id] = inf
            # 超出上限时淘汰最早未使用的项
            while len(_ollama_inferencer_cache) > _MAX_OLLAMA_CACHE:
                _ollama_inferencer_cache.popitem(last=False)
            return inf
    with _running_lock:
        entry = RUNNING_INSTANCES.get(run_id)
    if not entry:
        return None
    inf = entry.get("inferencer")
    if inf is None:
        raise EngineNotRunningError("该模型正在启动中（如拉取/加载），请稍候再试")
    return inf
