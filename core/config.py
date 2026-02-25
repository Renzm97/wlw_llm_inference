# -*- coding: utf-8 -*-
"""
配置加载、模型目录与内置模型列表。
从项目根目录或环境变量 CONFIG_PATH 加载 config.json，并初始化 OLLAMA_MODELS、HUGGINGFACE_HUB_CACHE。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.config")


def _project_root() -> str:
    """项目根目录（与 main.py 同级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config() -> Dict[str, Any]:
    """从项目目录或环境变量 CONFIG_PATH 指定路径加载 config.json。"""
    default: Dict[str, Any] = {
        "default_model_name": "llama3.2",
        "models_dir": "./models",
        "models_subdir_ollama": "ollama",
        "models_subdir_hf": "HF",
        "hf_token": None,
        "ollama": {"base_url": "http://localhost:11434"},
        "vllm": {
            "base_url": None,
            "local_model_path": None,
            "model_aliases": {"llama3.2": "Qwen/Qwen2-0.5B-Instruct"},
            "gpu_memory_utilization": 0.65,
        },
        "sglang": {"base_url": "http://localhost:30000"},
    }
    config_path = os.environ.get("CONFIG_PATH")
    if not config_path:
        config_path = os.path.join(_project_root(), "config.json")
    if not os.path.isfile(config_path):
        return default
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for key in default:
            if key not in loaded:
                loaded[key] = default[key]
            elif isinstance(default.get(key), dict) and isinstance(loaded.get(key), dict):
                for sub in default[key]:
                    if sub not in loaded[key]:
                        loaded[key][sub] = default[key][sub]
        return loaded
    except Exception as e:
        logger.warning("加载 config.json 失败，使用默认配置: %s", e)
        return default


CONFIG: Dict[str, Any] = _load_config()


def _setup_models_dirs() -> None:
    """根据配置创建模型目录并设置 OLLAMA_MODELS、HUGGINGFACE_HUB_CACHE。"""
    project_root = _project_root()
    models_base = CONFIG.get("models_dir") or "./models"
    if not os.path.isabs(models_base):
        models_base = os.path.join(project_root, models_base)
    models_base = os.path.normpath(os.path.abspath(models_base))
    ollama_subdir = CONFIG.get("models_subdir_ollama") or "ollama"
    hf_subdir = CONFIG.get("models_subdir_hf") or "HF"
    ollama_path = os.path.join(models_base, ollama_subdir)
    hf_path = os.path.join(models_base, hf_subdir)
    os.makedirs(ollama_path, exist_ok=True)
    os.makedirs(hf_path, exist_ok=True)
    os.environ["OLLAMA_MODELS"] = ollama_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_path
    logger.info(
        "模型目录: 根目录=%s, Ollama=%s, HF=%s",
        models_base, ollama_path, hf_path,
    )


def _setup_hf_token() -> None:
    token = CONFIG.get("hf_token")
    if token and isinstance(token, str) and token.strip():
        os.environ["HF_TOKEN"] = token.strip()
        logger.info("已从 config 设置 HF_TOKEN（用于 gated/私有模型）")


def get_platform_models_dir() -> str:
    """平台模型根目录。"""
    project_root = _project_root()
    models_base = CONFIG.get("models_dir") or "./models"
    if not os.path.isabs(models_base):
        models_base = os.path.join(project_root, models_base)
    return os.path.normpath(os.path.abspath(models_base))


def get_platform_hf_dir() -> str:
    """HuggingFace 缓存目录。"""
    base = get_platform_models_dir()
    hf_subdir = CONFIG.get("models_subdir_hf") or "HF"
    return os.path.join(base, hf_subdir)


# 导入时执行目录与 token 初始化
_setup_models_dirs()
_setup_hf_token()


# ---------- 模型目录（models.json）----------
def _load_models_catalog() -> List[Dict[str, Any]]:
    """从项目根目录或环境变量 MODELS_CONFIG 加载 models.json。"""
    config_path = os.environ.get("MODELS_CONFIG")
    if not config_path:
        config_path = os.path.join(_project_root(), "models.json")
    if not os.path.isfile(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models") or []
    except Exception as e:
        logger.warning("加载 models.json 失败: %s", e)
        return []


MODELS_CATALOG: List[Dict[str, Any]] = _load_models_catalog()


def get_models_catalog() -> List[Dict[str, Any]]:
    """返回模型目录（供 API 与前端使用），每项含 id, name, description, official_url, sizes, quantizations, engines, formats。"""
    return MODELS_CATALOG


def get_model_variant(model_id: str, size: str) -> Optional[Dict[str, Any]]:
    """
    根据 model_id 与 size 解析出对应 variant（hf_repo、ollama_name 等）。
    若未配置 models.json 或未匹配到则返回 None。
    """
    if not size or not MODELS_CATALOG:
        return None
    model = next((m for m in MODELS_CATALOG if m.get("id") == model_id), None)
    if not model:
        return None
    sizes = model.get("sizes") or []
    variant = next((s for s in sizes if str(s.get("size")) == str(size)), None)
    if not variant:
        return None
    return {
        "model_id": model_id,
        "name": model.get("name", model_id),
        "size": variant.get("size"),
        "hf_repo": variant.get("hf_repo"),
        "ollama_name": variant.get("ollama_name") or model_id,
        "quantizations": model.get("quantizations") or ["none"],
        "engines": model.get("engines") or ["ollama", "vllm", "sglang"],
        "official_url": model.get("official_url") or (f"https://huggingface.co/{variant.get('hf_repo', '')}" if variant.get("hf_repo") else None),
    }


# 兼容旧逻辑：无 models.json 时使用的内置列表（单 variant  per 模型）
BUILTIN_MODELS: List[Dict[str, Any]] = []
if not MODELS_CATALOG:
    BUILTIN_MODELS = [
        {"id": "qwen2-0.5b", "name": "Qwen2 0.5B", "hf_repo": "Qwen/Qwen2-0.5B-Instruct", "official_url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct", "quantizations": ["none"], "engines": ["ollama", "vllm", "sglang"], "ollama_name": "qwen2:0.5b"},
        {"id": "llama3.2", "name": "Llama 3.2", "hf_repo": "meta-llama/Llama-3.2-1B-Instruct", "official_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct", "quantizations": ["none"], "engines": ["ollama", "vllm", "sglang"], "ollama_name": "llama3.2"},
    ]
else:
    # 从目录展开为扁平列表（每个 size 一条），兼容依赖 BUILTIN_MODELS 的代码
    for m in MODELS_CATALOG:
        for s in (m.get("sizes") or []):
            BUILTIN_MODELS.append({
                "id": f"{m['id']}-{s.get('size', '').replace('.', '-')}".strip("-"),
                "name": f"{m.get('name', m['id'])} {s.get('size', '')}".strip(),
                "hf_repo": s.get("hf_repo"),
                "official_url": m.get("official_url"),
                "quantizations": m.get("quantizations") or ["none"],
                "engines": m.get("engines") or ["ollama", "vllm", "sglang"],
                "ollama_name": s.get("ollama_name") or m.get("id"),
            })


def ensure_model_downloaded(model_id: str, size: Optional[str] = None) -> str:
    """
    确保模型已下载到平台模型目录，返回用于加载的 hf_repo。
    若配置了 models.json，则用 model_id + size 解析 variant 再按 hf_repo 下载；
    否则按旧逻辑用 model_id 在 BUILTIN_MODELS 中查找。
    """
    from core.exceptions import EngineNotInstalledError, ModelNotFoundError

    hf_repo: Optional[str] = None
    if MODELS_CATALOG and size:
        variant = get_model_variant(model_id, size)
        if variant and variant.get("hf_repo"):
            hf_repo = variant["hf_repo"]
    if not hf_repo and MODELS_CATALOG and not size:
        # 未传 size 时取该模型第一个 size
        model = next((m for m in MODELS_CATALOG if m.get("id") == model_id), None)
        if model and (model.get("sizes")):
            first = model["sizes"][0]
            hf_repo = first.get("hf_repo")
    if not hf_repo:
        entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
        if entry:
            hf_repo = entry.get("hf_repo")
    if not hf_repo:
        raise ModelNotFoundError(f"未知模型或未指定 size: model_id={model_id!r}, size={size!r}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise EngineNotInstalledError("请安装 huggingface_hub: pip install huggingface_hub")
    cache_dir = get_platform_hf_dir()
    os.makedirs(cache_dir, exist_ok=True)
    snapshot_download(repo_id=hf_repo, cache_dir=cache_dir)
    return hf_repo
