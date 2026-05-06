# -*- coding: utf-8 -*-
"""
配置加载、模型目录与内置模型列表。
从项目根目录或环境变量 CONFIG_PATH 加载 config.json，并初始化 OLLAMA_MODELS、HUGGINGFACE_HUB_CACHE。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
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
        "model_source": "huggingface",  # 可选: "huggingface" | "modelscope"
        "hf_token": None,
        "ms_token": None,
        "ollama": {"base_url": "http://localhost:11434"},
        "vllm": {
            "base_url": None,
            "local_model_path": None,
            "model_aliases": {},
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
        if not isinstance(loaded, dict):
            logger.warning("config.json 格式无效（期望字典），使用默认配置")
            return default
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
    # ModelScope 缓存目录与 HF 保持一致，方便统一管理
    os.environ["MODELSCOPE_CACHE"] = hf_path
    logger.info(
        "模型目录: 根目录=%s, Ollama=%s, HF=%s",
        models_base, ollama_path, hf_path,
    )


def _setup_tokens() -> None:
    model_source = CONFIG.get("model_source", "huggingface")
    if model_source == "modelscope":
        token = CONFIG.get("ms_token")
        if not token:
            token = os.environ.get("MODELSCOPE_API_TOKEN")
        if token and isinstance(token, str) and token.strip():
            os.environ["MODELSCOPE_API_TOKEN"] = token.strip()
            logger.info("已设置 MODELSCOPE_API_TOKEN（用于 ModelScope 私有模型）")
    else:
        token = CONFIG.get("hf_token")
        if not token:
            token = os.environ.get("HF_TOKEN")
        if token and isinstance(token, str) and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            logger.info("已设置 HF_TOKEN（用于 gated/私有模型）")


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
_setup_tokens()


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
        "ms_repo": variant.get("ms_repo") or variant.get("hf_repo"),
        "ollama_name": variant.get("ollama_name") or model_id,
        "quantizations": model.get("quantizations") or ["none"],
        "quantization_repos": variant.get("quantization_repos") or {},
        "engines": model.get("engines") or ["vllm", "ollama", "sglang"],
        "official_url": model.get("official_url") or (f"https://huggingface.co/{variant.get('hf_repo', '')}" if variant.get("hf_repo") else None),
    }


# 兼容旧逻辑：无 models.json 时使用的内置列表（单 variant  per 模型）
BUILTIN_MODELS: List[Dict[str, Any]] = []
if not MODELS_CATALOG:
    BUILTIN_MODELS = [
        {"id": "qwen2-0.5b", "name": "Qwen2 0.5B", "hf_repo": "Qwen/Qwen2-0.5B-Instruct", "ms_repo": "Qwen/Qwen2-0.5B-Instruct", "official_url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct", "quantizations": ["none"], "engines": ["vllm", "ollama", "sglang"], "ollama_name": "qwen2:0.5b"},
        {"id": "llama3.2", "name": "Llama 3.2", "hf_repo": "meta-llama/Llama-3.2-1B-Instruct", "ms_repo": "LLM-Research/Llama-3.2-1B-Instruct", "official_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct", "quantizations": ["none"], "engines": ["vllm", "ollama", "sglang"], "ollama_name": "llama3.2"},
    ]
# 当 MODELS_CATALOG 存在时，BUILTIN_MODELS 保持为空，避免 ID 命名空间冲突
# 依赖 BUILTIN_MODELS 的代码应优先使用 get_model_variant() 或搜索 MODELS_CATALOG


def _verify_model_dir(local_dir: str) -> None:
    """
    验证模型目录完整性，防止 snapshot_download 返回成功但实际文件缺失
    （ModelScope 在个别文件下载失败时仍可能正常返回）。
    检查项：
      1. 不存在明确的下载中断标记文件（如 .incomplete、*.tmp、*.part）
      2. 若存在 model.safetensors.index.json 或 pytorch_model.bin.index.json，
         则解析并确认所有引用的权重文件均存在
      3. 若无索引文件，至少应存在一个权重文件（.safetensors/.bin/.pth/.ckpt）
    验证通过则静默返回，否则抛出 RuntimeError。
    """
    # 1) 检查明确的下载中断标记（只检查文件，不检查 ModelScope/HF 的正常缓存目录如 ._____temp）
    for root, dirs, files in os.walk(local_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith((".incomplete", ".tmp", ".temp", ".part", ".download")):
                raise RuntimeError(f"发现未完成的下载标记文件: {os.path.join(root, name)}")
        # 避免深入隐藏目录（如 .git、._____temp），它们不影响模型加载
        dirs[:] = [d for d in dirs if not d.startswith(".")]

    # 2) 尝试通过索引文件校验权重完整性
    index_files = [
        os.path.join(local_dir, "model.safetensors.index.json"),
        os.path.join(local_dir, "pytorch_model.bin.index.json"),
        os.path.join(local_dir, "model.ckpt.index.json"),
    ]
    weight_found = False
    for idx_path in index_files:
        if not os.path.isfile(idx_path):
            continue
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"索引文件损坏或无法解析: {idx_path}: {e}")
        weight_map = idx_data.get("weight_map") if isinstance(idx_data, dict) else None
        if not isinstance(weight_map, dict):
            continue
        missing = []
        for filename in set(weight_map.values()):
            fpath = os.path.join(local_dir, filename)
            if not os.path.isfile(fpath):
                missing.append(filename)
        if missing:
            raise RuntimeError(f"索引文件引用的权重文件缺失 ({len(missing)} 个): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        weight_found = True
        break

    if not weight_found:
        # 3) 无索引文件时，至少需要一个常见权重文件
        has_weight = False
        for entry in os.listdir(local_dir):
            if entry.lower().endswith((".safetensors", ".bin", ".pth", ".ckpt")):
                has_weight = True
                break
        if not has_weight:
            raise RuntimeError("目录中未找到任何模型权重文件（.safetensors/.bin/.pth/.ckpt）")


def ensure_model_downloaded(model_id: str, size: Optional[str] = None, quantization: Optional[str] = None) -> str:
    """
    确保模型已下载到平台模型目录，返回本地绝对路径（供 vLLM/SGLang 直接加载）。
    若配置了 models.json，则用 model_id + size 解析 variant 再按仓库名下载；
    否则按旧逻辑用 model_id 在 BUILTIN_MODELS 中查找。
    当 quantization 非 none 且 models.json 中配置了 quantization_repos 时，
    将下载对应的量化版模型并返回量化标识子目录。
    下载源由 config.json 中的 model_source 控制（"huggingface" 或 "modelscope"）。
    当 source 为 modelscope 时会优先使用 ms_repo，若未配置则回退到 hf_repo。
    """
    from core.exceptions import EngineNotInstalledError, EngineNotRunningError, InvalidParameterError, ModelNotFoundError

    hf_repo: Optional[str] = None
    ms_repo: Optional[str] = None

    if MODELS_CATALOG and size:
        variant = get_model_variant(model_id, size)
        if variant and variant.get("hf_repo"):
            hf_repo = variant["hf_repo"]
            ms_repo = variant.get("ms_repo") or hf_repo
            # 处理量化模型仓库映射（size 级别）
            if quantization and quantization != "none":
                q_repos = variant.get("quantization_repos") or {}
                q_cfg = q_repos.get(quantization)
                if q_cfg and q_cfg.get("hf_repo"):
                    hf_repo = q_cfg["hf_repo"]
                    ms_repo = q_cfg.get("ms_repo") or hf_repo
                else:
                    raise ModelNotFoundError(
                        f"模型 {model_id} 的 {size} 版本未配置 {quantization} 量化模型仓库，"
                        f"请在 models.json 对应 size 的 quantization_repos 中补充 hf_repo"
                    )
    if not hf_repo and MODELS_CATALOG and not size:
        # 未传 size 时取该模型第一个 size
        model = next((m for m in MODELS_CATALOG if m.get("id") == model_id), None)
        if model and (model.get("sizes")):
            first = model["sizes"][0]
            hf_repo = first.get("hf_repo")
            ms_repo = first.get("ms_repo") or hf_repo
            if quantization and quantization != "none":
                q_repos = first.get("quantization_repos") or {}
                q_cfg = q_repos.get(quantization)
                if q_cfg and q_cfg.get("hf_repo"):
                    hf_repo = q_cfg["hf_repo"]
                    ms_repo = q_cfg.get("ms_repo") or hf_repo
                else:
                    raise ModelNotFoundError(
                        f"模型 {model_id} 未配置 {quantization} 量化模型仓库"
                    )
    if not hf_repo:
        entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
        if entry:
            hf_repo = entry.get("hf_repo")
            ms_repo = entry.get("ms_repo") or hf_repo
    if not hf_repo:
        raise ModelNotFoundError(f"未知模型或未指定 size: model_id={model_id!r}, size={size!r}")

    # 防止路径遍历：model_id 来自用户输入，需过滤非法字符
    if ".." in model_id or "/" in model_id or "\\" in model_id:
        raise InvalidParameterError(f"model_id 包含非法字符: {model_id!r}")

    model_source = CONFIG.get("model_source", "huggingface")
    repo_id = ms_repo if model_source == "modelscope" else hf_repo

    if model_source == "modelscope" and not ms_repo:
        logger.warning("当前 model_source=modelscope，但模型 %s 未配置 ms_repo，将回退使用 hf_repo=%s", model_id, hf_repo)
        repo_id = hf_repo

    # 使用 model_id 作为本地扁平目录名；若指定了量化，则使用 model_id-quantization 子目录
    local_dir_name = f"{model_id}-{quantization}" if (quantization and quantization != "none") else model_id
    local_dir = os.path.join(get_platform_models_dir(), local_dir_name)
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        try:
            _verify_model_dir(local_dir)
            logger.info("模型 %s 已存在于本地目录且校验通过: %s", model_id, local_dir)
            return local_dir
        except RuntimeError as verify_err:
            logger.warning("模型 %s 本地目录校验失败，将重新下载: %s", model_id, verify_err)
            # 尝试清理损坏的目录后重新下载
            try:
                shutil.rmtree(local_dir)
            except Exception:
                pass
    os.makedirs(local_dir, exist_ok=True)

    # 先检查依赖是否安装，避免被外层异常捕获
    snapshot_download = None
    if model_source == "modelscope":
        try:
            from modelscope import snapshot_download as _ms_download
            snapshot_download = _ms_download
        except ImportError:
            raise EngineNotInstalledError("请安装 modelscope: pip install modelscope")
    else:
        try:
            from huggingface_hub import snapshot_download as _hf_download
            snapshot_download = _hf_download
        except ImportError:
            raise EngineNotInstalledError("请安装 huggingface_hub: pip install huggingface_hub")

    local_path: str = ""
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            if model_source == "modelscope":
                logger.info("使用 ModelScope 下载模型: %s -> %s (第 %s 次尝试)", repo_id, local_dir, attempt)
                local_path = snapshot_download(model_id=repo_id, local_dir=local_dir)
            else:
                logger.info("使用 HuggingFace 下载模型: %s -> %s (第 %s 次尝试)", repo_id, local_dir, attempt)
                local_path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
            # snapshot_download 可能返回成功但部分文件缺失，必须校验
            _verify_model_dir(local_path if local_path else local_dir)
            break
        except Exception as e:
            last_exc = e
            logger.warning("下载模型 %s 第 %s 次尝试失败: %s", repo_id, attempt, e)
            # 清理临时目录，避免残留影响下次重试
            for temp_name in ("._____temp", ".huggingface", ".tmp"):
                temp_path = os.path.join(local_dir, temp_name)
                if os.path.exists(temp_path):
                    try:
                        shutil.rmtree(temp_path)
                    except Exception:
                        pass
            if attempt < 3:
                time.sleep(3)
            else:
                break

    if last_exc is not None:
        err_type = type(last_exc).__name__
        err_msg = str(last_exc).lower()
        # 清理空目录，避免下次误判为已下载
        try:
            if os.path.isdir(local_dir) and not os.listdir(local_dir):
                os.rmdir(local_dir)
        except Exception:
            pass
        if err_type == "GatedRepoError" or "gated repo" in err_msg:
            raise ModelNotFoundError(
                f"模型 {repo_id} 为 HuggingFace 受限仓库（gated），"
                f"请先访问 https://huggingface.co/{repo_id} 接受使用条款，"
                f"并在 config.json 中配置 hf_token 或设置 HF_TOKEN 环境变量后重试"
            ) from last_exc
        if err_type == "RepositoryNotFoundError" or "not found" in err_msg:
            raise ModelNotFoundError(
                f"模型仓库 {repo_id} 在 HuggingFace/ModelScope 上不存在，请检查 model_id 与 size 配置"
            ) from last_exc
        raise EngineNotRunningError(f"下载模型 {repo_id} 失败: {last_exc}") from last_exc
    return local_path if local_path else local_dir
