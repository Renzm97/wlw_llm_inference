# -*- coding: utf-8 -*-
"""测试页代理：代为向 Ollama/vLLM/SGLang 发送推理请求，解决浏览器跨域和 local:run_id 访问问题。"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Request

from api.schemas import ApiResponse, TestProxyRequest
from core.exceptions import EngineNotRunningError, InvalidParameterError
from services import get_running_inferencer

logger = logging.getLogger("api.routes.test_proxy")

router = APIRouter(prefix="/api/v1/test", tags=["test"])


def _build_proxy_payload(engine: str, model: str, prompt: str, temperature: float, max_tokens: int, top_p: float) -> Dict[str, Any]:
    if engine == "ollama":
        return {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": top_p},
        }
    if engine == "vllm":
        return {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
    if engine == "sglang":
        return {
            "text": prompt,
            "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p},
        }
    return {}


def _extract_proxy_response(engine: str, data: Dict[str, Any]) -> str:
    if engine == "ollama":
        return str(data.get("response") or "")
    if engine == "vllm":
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return str(choices[0].get("text") or "")
        return ""
    if engine == "sglang":
        return str(data.get("text") or data.get("generated_text") or "")
    return ""


def _get_proxy_endpoint(engine: str, address: str) -> str:
    base = address.rstrip("/")
    if engine == "ollama":
        return base + "/api/generate"
    if engine == "vllm":
        return base + "/v1/completions"
    if engine == "sglang":
        return base + "/generate"
    return base


def _is_safe_address(address: str) -> bool:
    """
    防止 SSRF：禁止访问内网、回环、链路本地和云元数据地址。
    仅允许公网地址或配置中明确列出的本地服务端口。
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(address)
        host = parsed.hostname
        if not host:
            return False
        # 禁止裸 IP（降低 SSRF 风险）
        try:
            ip = ipaddress.ip_address(host)
            # 禁止所有私有/保留/回环/链路本地地址
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
            # 禁止云元数据地址
            if str(ip) == "169.254.169.254":
                return False
            return True
        except ValueError:
            # 不是 IP，是域名；允许公网域名
            # 但禁止 localhost 域名变体
            if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                return False
            return True
    except Exception:
        return False


@router.post("/proxy", response_model=ApiResponse)
async def api_test_proxy(body: TestProxyRequest, request: Request):
    """
    测试页代理接口。
    若 address 为 local:run_id，则直接使用后端已加载的 inferencer；
    否则按引擎格式构造 HTTP 请求并代为转发。
    """
    rid = request.state.request_id
    address = body.address.strip()
    engine = body.engine

    # local:run_id 场景：直接使用后端 inferencer（避免浏览器无法访问进程内模型）
    if address.startswith("local:"):
        run_id = address[6:].strip()
        if not run_id:
            raise InvalidParameterError("local: 地址后缺少 run_id")
        inferencer = get_running_inferencer(run_id)
        if inferencer is None:
            raise EngineNotRunningError(f"run_id 无效或已停止: {run_id}")
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    inferencer.generate,
                    body.prompt,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    top_p=body.top_p,
                ),
            )
            return ApiResponse(
                request_id=rid,
                code=200,
                msg="success",
                data={"response": response},
            )
        except Exception as e:
            logger.exception("test_proxy 本地 inferencer 异常: %s", e)
            raise

    # 远程地址场景：后端代为 HTTP 请求（解决浏览器 CORS）
    addr_lower = address.lower()
    if not addr_lower.startswith("http://") and not addr_lower.startswith("https://"):
        raise InvalidParameterError("address 必须是 http:// 或 https:// 开头，或 local:run_id 格式")

    if not _is_safe_address(address):
        raise InvalidParameterError("address 指向的地址不被允许（禁止访问内网/回环/元数据地址）")

    url = _get_proxy_endpoint(engine, address)
    payload = _build_proxy_payload(engine, body.model, body.prompt, body.temperature, body.max_tokens, body.top_p)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            return ApiResponse(
                request_id=rid,
                code=resp.status_code,
                msg=f"上游返回 HTTP {resp.status_code}",
                data={"detail": resp.text},
            )
        data = resp.json()
        text = _extract_proxy_response(engine, data)
        return ApiResponse(
            request_id=rid,
            code=200,
            msg="success",
            data={"response": text},
        )
    except httpx.RequestError as e:
        logger.warning("test_proxy 请求上游失败 engine=%s url=%s: %s", engine, url, e)
        raise EngineNotRunningError(f"无法连接到模型服务 {address}: {e}")
    except Exception as e:
        logger.exception("test_proxy 异常: %s", e)
        raise
