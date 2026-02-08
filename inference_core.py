#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理核心模块：引擎适配、加载与推理，以及「验证启动的模型是否可用」。
供 llm_inference 启动模型后做一次短文本生成以校验模型可用；也可被直接导入做推理。
"""

from __future__ import annotations

import abc
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

try:
    import sglang
    from sglang import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    sglang = None
    Engine = None
    SGLANG_AVAILABLE = False


logger = logging.getLogger("inference_core")


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
        base = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base, "config.json")
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
    project_root = os.path.dirname(os.path.abspath(__file__))
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
_setup_models_dirs()
_setup_hf_token()


def _get_platform_models_dir() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_base = CONFIG.get("models_dir") or "./models"
    if not os.path.isabs(models_base):
        models_base = os.path.join(project_root, models_base)
    return os.path.normpath(os.path.abspath(models_base))


def _get_platform_hf_dir() -> str:
    base = _get_platform_models_dir()
    hf_subdir = CONFIG.get("models_subdir_hf") or "HF"
    return os.path.join(base, hf_subdir)


BUILTIN_MODELS: List[Dict[str, Any]] = [
    {
        "id": "qwen2-0.5b",
        "name": "Qwen2 0.5B",
        "hf_repo": "Qwen/Qwen2-0.5B-Instruct",
        "official_url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct",
        "quantizations": ["none"],
        "engines": ["ollama", "vllm", "sglang"],
        "ollama_name": "qwen2:0.5b",
    },
    {
        "id": "llama3.2",
        "name": "Llama 3.2",
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "official_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        "quantizations": ["none"],
        "engines": ["ollama", "vllm", "sglang"],
        "ollama_name": "llama3.2",
    },
]


def _ensure_model_downloaded(model_id: str) -> str:
    """确保模型已下载到平台模型目录（配置文件中的路径），返回用于加载的 repo_id。"""
    entry = next((m for m in BUILTIN_MODELS if m["id"] == model_id), None)
    if not entry:
        raise ModelNotFoundError(f"未知内置模型: {model_id}")
    hf_repo = entry["hf_repo"]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise EngineNotInstalledError("请安装 huggingface_hub: pip install huggingface_hub")
    cache_dir = _get_platform_hf_dir()
    os.makedirs(cache_dir, exist_ok=True)
    # 仅传 repo_id / cache_dir，兼容新旧版 huggingface_hub（新版可能已移除 local_files_ok）
    snapshot_download(repo_id=hf_repo, cache_dir=cache_dir)
    return hf_repo


# ---------- 异常 ----------
class LLMInferenceError(Exception):
    pass


class EngineNotInstalledError(LLMInferenceError):
    pass


class EngineNotRunningError(LLMInferenceError):
    pass


class ModelNotFoundError(LLMInferenceError):
    pass


class InvalidParameterError(LLMInferenceError):
    pass


class StructuredOutputNotSupportedError(LLMInferenceError):
    pass


# ---------- 引擎适配基类 ----------
class BaseLLMAdapter(abc.ABC):
    @property
    @abc.abstractmethod
    def engine_type(self) -> str:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    @abc.abstractmethod
    def check_service(self, model_name: str) -> None:
        pass

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        pass

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        pass

    def structured_generate(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        *,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], str]:
        return self.generate(
            prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


# ---------- Ollama ----------
class OllamaAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @property
    def engine_type(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        if OLLAMA_AVAILABLE:
            return True
        try:
            with httpx.Client(timeout=2.0) as c:
                return c.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def check_service(self, model_name: str) -> None:
        if not OLLAMA_AVAILABLE:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(f"{self.base_url}/api/tags")
                    if r.status_code != 200:
                        raise EngineNotRunningError(f"Ollama 服务未就绪: {self.base_url}")
                    data = r.json()
                    names = [m.get("name") for m in data.get("models", [])]
                    if model_name not in names and not any(m.startswith(model_name) or model_name in m for m in names):
                        raise ModelNotFoundError(f"模型 '{model_name}' 不在 Ollama 已拉取列表中。可用: {names}。")
            except httpx.RequestError as e:
                raise EngineNotRunningError(f"无法连接 Ollama: {e}")
            return
        try:
            ollama.list()
        except Exception as e:
            raise EngineNotRunningError(f"Ollama 服务未就绪: {e}")
        models = [(m.get("model") or m.get("name") or "") for m in ollama.list().get("models", [])]
        if model_name not in models and not any(m.startswith(model_name) or model_name in m for m in models):
            raise ModelNotFoundError(f"模型 '{model_name}' 不存在。可用: {models}。")

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        if OLLAMA_AVAILABLE:
            opts = {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p, **{k: v for k, v in kwargs.items() if v is not None}}
            resp = ollama.generate(model=model_name, prompt=prompt, options=opts)
            return resp.get("response", "").strip()
        payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p}}
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        if OLLAMA_AVAILABLE:
            opts = {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p, **{k: v for k, v in kwargs.items() if v is not None}}
            resp = ollama.chat(model=model_name, messages=[{"role": m["role"], "content": m["content"]} for m in messages], options=opts)
            return resp.get("message", {}).get("content", "").strip()
        payload = {"model": model_name, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p}}
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()


# ---------- VLLM ----------
def _vllm_resolve_path(model_name: str, local_model_path: Optional[str]) -> str:
    def _expand(s: str) -> str:
        return os.path.expanduser(s) if (s.startswith("~") or s.startswith("/")) else s
    if local_model_path:
        return _expand(local_model_path)
    vllm_cfg = CONFIG.get("vllm") or {}
    if vllm_cfg.get("local_model_path"):
        return _expand(vllm_cfg["local_model_path"])
    raw = (vllm_cfg.get("model_aliases") or {}).get(model_name, model_name)
    return _expand(raw)


class VLLMAdapter(BaseLLMAdapter):
    def __init__(self, base_url: Optional[str] = None, local_model_path: Optional[str] = None, gpu_memory_utilization: Optional[float] = None):
        self.base_url = base_url
        self.local_model_path = local_model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None

    @property
    def engine_type(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        return VLLM_AVAILABLE

    def _get_llm(self, model_name: str) -> Any:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm。请执行: pip install vllm")
        if self.base_url:
            return None
        path = _vllm_resolve_path(model_name, self.local_model_path)
        if self._llm is None:
            vllm_cfg = CONFIG.get("vllm") or {}
            gpu_util = self._gpu_memory_utilization if self._gpu_memory_utilization is not None else vllm_cfg.get("gpu_memory_utilization")
            kwargs: Dict[str, Any] = {"model": path, "trust_remote_code": True}
            if gpu_util is not None:
                kwargs["gpu_memory_utilization"] = float(gpu_util)
            self._llm = LLM(**kwargs)
        return self._llm

    def check_service(self, model_name: str) -> None:
        if not VLLM_AVAILABLE:
            raise EngineNotInstalledError("未安装 vllm")
        if self.base_url:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(f"{self.base_url.replace('/v1', '')}/health")
                    if r.status_code != 200:
                        raise EngineNotRunningError(f"VLLM 服务未就绪: {self.base_url}")
            except httpx.RequestError as e:
                raise EngineNotRunningError(f"无法连接 VLLM: {e}")
        else:
            self._get_llm(model_name)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
        parts.append("Assistant: ")
        return "\n".join(parts)

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        if self.base_url:
            with httpx.Client(timeout=120.0) as c:
                r = c.post(f"{self.base_url}/completions", json={"model": model_name, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p})
                r.raise_for_status()
                choices = r.json().get("choices", [])
                return (choices[0].get("text", "") if choices else "").strip()
        llm = self._get_llm(model_name)
        outs = llm.generate([prompt], sampling)
        return (outs[0].outputs[0].text if outs and outs[0].outputs else "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)


# ---------- SGLang ----------
class SGLangAdapter(BaseLLMAdapter):
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url.rstrip("/")
        self._engine: Any = None

    @property
    def engine_type(self) -> str:
        return "sglang"

    def is_available(self) -> bool:
        return SGLANG_AVAILABLE

    def check_service(self, model_name: str) -> None:
        if not SGLANG_AVAILABLE:
            raise EngineNotInstalledError("未安装 sglang")
        try:
            with httpx.Client(timeout=5.0) as c:
                if c.get(f"{self.base_url}/get_model_info").status_code != 200:
                    raise EngineNotRunningError(f"SGLang 服务未就绪: {self.base_url}")
        except httpx.RequestError as e:
            raise EngineNotRunningError(f"无法连接 SGLang: {e}")

    def generate(self, prompt: str, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self.check_service(model_name)
        payload = {"text": prompt, "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}}
        with httpx.Client(timeout=120.0) as c:
            r = c.post(f"{self.base_url}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("text", "") or data.get("generated_text", "") or "").strip()

    def chat(self, messages: List[Dict[str, str]], *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        parts.append("assistant: ")
        return self.generate("\n".join(parts), model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def structured_generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, *, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> Union[Dict[str, Any], str]:
        self.check_service(model_name)
        schema_hint = f"\n请严格按以下 JSON 结构返回：\n{json.dumps(schema, ensure_ascii=False)}\n" if schema else ""
        full_prompt = f"{prompt}{schema_hint}\n请只输出合法 JSON，不要其他文字。"
        raw = self.generate(full_prompt, model_name=model_name, temperature=max(0.1, temperature), max_tokens=max_tokens, top_p=top_p, **kwargs)
        try:
            raw = raw.strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


# ---------- 核心推理类 ----------
class LLMInferencer:
    ENGINE_MAP = {"ollama": OllamaAdapter, "vllm": VLLMAdapter, "sglang": SGLangAdapter}

    def __init__(
        self,
        engine_type: Literal["ollama", "vllm", "sglang"],
        model_name: Optional[str] = None,
        *,
        ollama_base_url: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        vllm_local_model_path: Optional[str] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        sglang_base_url: Optional[str] = None,
    ):
        if engine_type not in self.ENGINE_MAP:
            raise InvalidParameterError(f"engine_type 仅允许 ollama/vllm/sglang，当前: {engine_type}")
        self.engine_type = engine_type
        self.model_name = model_name or CONFIG.get("default_model_name", "llama3.2")
        adapter_cls = self.ENGINE_MAP[engine_type]
        if engine_type == "ollama":
            url = ollama_base_url or (CONFIG.get("ollama") or {}).get("base_url", "http://localhost:11434")
            self._adapter: BaseLLMAdapter = adapter_cls(base_url=url)
        elif engine_type == "vllm":
            vllm_cfg = CONFIG.get("vllm") or {}
            url = vllm_base_url if vllm_base_url is not None else vllm_cfg.get("base_url")
            path = vllm_local_model_path if vllm_local_model_path is not None else vllm_cfg.get("local_model_path")
            self._adapter = adapter_cls(base_url=url, local_model_path=path, gpu_memory_utilization=vllm_gpu_memory_utilization)
        else:
            url = sglang_base_url or (CONFIG.get("sglang") or {}).get("base_url", "http://localhost:30000")
            self._adapter = adapter_cls(base_url=url)
        if not self._adapter.is_available():
            raise EngineNotInstalledError(f"引擎 '{engine_type}' 依赖未安装")
        self._adapter.check_service(self.model_name)
        logger.info("LLMInferencer 初始化成功: engine=%s, model=%s", engine_type, self.model_name)

    def _validate_common(self, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95) -> None:
        if not (0 <= temperature <= 2):
            raise InvalidParameterError("temperature 需在 [0, 2] 范围内")
        if max_tokens < 1:
            raise InvalidParameterError("max_tokens 需为正整数")
        if not (0 <= top_p <= 1):
            raise InvalidParameterError("top_p 需在 [0, 1] 范围内")

    def generate(self, prompt: str, *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.generate(prompt, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def chat(self, messages: List[Dict[str, str]], *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> str:
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.chat(messages, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)

    def structured_generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, *, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, **kwargs: Any) -> Union[Dict[str, Any], str]:
        if self.engine_type != "sglang":
            raise StructuredOutputNotSupportedError("structured_generate 仅支持 SGLang")
        self._validate_common(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        model = model_name or self.model_name
        self._adapter.check_service(model)
        return self._adapter.structured_generate(prompt, schema, model_name=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)


def validate_model_usable(inferencer: LLMInferencer, max_tokens: int = 5) -> bool:
    """验证已启动的模型是否可用：执行一次短文本生成，成功返回 True，异常返回 False。"""
    try:
        inferencer.generate("hi", max_tokens=max_tokens)
        return True
    except Exception as e:
        logger.warning("验证模型可用性失败: %s", e)
        return False
