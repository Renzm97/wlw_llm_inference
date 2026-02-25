#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：启动 API 服务或运行测试。
  - 仅运行核心推理测试: python main.py
  - 启动 API 服务: python main.py --serve [--host 0.0.0.0] [--port 8000]
  - API 接口测试: python main.py --api-test [--port 8000]
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def run_core_tests() -> None:
    """运行核心推理测试：单轮、多轮、引擎检测。"""
    from core import (
        LLMInferencer,
        EngineNotInstalledError,
        EngineNotRunningError,
        ModelNotFoundError,
        StructuredOutputNotSupportedError,
    )

    test_engine: Optional[str] = "vllm"
    logger.info("========== 开始核心推理模块测试 ==========")

    engines = ("ollama", "vllm", "sglang")
    if test_engine:
        engines = (test_engine,)
        logger.info("当前仅测试引擎: %s", test_engine)

    reusable_inf: Optional[LLMInferencer] = None
    logger.info("1. 引擎可用性检查")
    for eng in engines:
        try:
            inf = LLMInferencer(engine_type=eng, model_name="llama3.2")
            if test_engine:
                reusable_inf = inf
            logger.info("  %s: 可用", eng)
        except EngineNotInstalledError as e:
            logger.info("  %s: 未安装 -> %s", eng, e)
        except (EngineNotRunningError, ModelNotFoundError, OSError) as e:
            logger.info("  %s: 服务/模型异常 -> %s", eng, e)

    run_eng = test_engine or "ollama"
    try:
        inf = reusable_inf if (reusable_inf and test_engine) else LLMInferencer(engine_type=run_eng, model_name="llama3.2")
        out = inf.generate("你好，请用一句话介绍你自己。", max_tokens=64)
        logger.info("2. 单轮推理(%s) 成功: %s", run_eng, out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("2. 单轮推理 跳过或失败: %s", e)

    try:
        inf = reusable_inf if (reusable_inf and test_engine) else LLMInferencer(engine_type=run_eng, model_name="llama3.2")
        msgs = [
            {"role": "user", "content": "我叫小明。"},
            {"role": "assistant", "content": "你好小明！"},
            {"role": "user", "content": "你还记得我叫什么吗？"},
        ]
        out = inf.chat(msgs, max_tokens=64)
        logger.info("3. 多轮对话(%s) 成功: %s", run_eng, out[:80] + "..." if len(out) > 80 else out)
    except Exception as e:
        logger.info("3. 多轮对话 跳过或失败: %s", e)

    if not test_engine or test_engine == "sglang":
        try:
            inf = LLMInferencer(engine_type="sglang", model_name="llama3.2")
            result = inf.structured_generate(
                "请输出一个包含 name 和 age 的 JSON。",
                schema={"name": "string", "age": "number"},
                max_tokens=128,
            )
            logger.info("4. 结构化输出(SGLang) 成功: %s", result)
        except StructuredOutputNotSupportedError:
            logger.info("4. 结构化输出 仅支持 SGLang，已跳过")
        except Exception as e:
            logger.info("4. 结构化输出 跳过或失败: %s", e)
    else:
        logger.info("4. 结构化输出 已跳过（当前仅测试 %s）", run_eng)

    logger.info("========== 核心推理模块测试结束 ==========")


def run_api_tests(base_url: str = "http://127.0.0.1:8000") -> None:
    """调用 FastAPI 接口的简单测试（需先启动服务）。"""
    try:
        import requests
    except ImportError:
        logger.warning("未安装 requests，跳过 API 测试")
        return
    logger.info("========== API 接口测试（base_url=%s）==========", base_url)
    headers = {"Content-Type": "application/json"}

    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=10)
        j = r.json()
        logger.info("GET /api/v1/models -> code=%s models_count=%s", j.get("code"), len((j.get("data") or {}).get("models") or []))
    except Exception as e:
        logger.info("GET /api/v1/models 请求失败: %s", e)

    try:
        r = requests.get(f"{base_url}/api/v1/models/running", timeout=10)
        j = r.json()
        logger.info("GET /api/v1/models/running -> code=%s running_count=%s", j.get("code"), len((j.get("data") or {}).get("running") or []))
    except Exception as e:
        logger.info("GET /api/v1/models/running 请求失败: %s", e)

    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/generate",
            json={"prompt": "你好", "engine_type": "ollama", "model_name": "llama3.2", "max_tokens": 32},
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info("POST /api/v1/llm/generate -> code=%s data=%s", j.get("code"), j.get("data"))
    except Exception as e:
        logger.info("API generate 请求失败: %s", e)

    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/chat",
            json={
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "1+1=?"},
                ],
                "engine_type": "ollama",
                "model_name": "llama3.2",
                "max_tokens": 32,
            },
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info("POST /api/v1/llm/chat -> code=%s data=%s", j.get("code"), j.get("data"))
    except Exception as e:
        logger.info("API chat 请求失败: %s", e)

    try:
        r = requests.post(
            f"{base_url}/api/v1/llm/structured-generate",
            json={
                "prompt": "输出一个 JSON：{\"name\": \"张三\", \"age\": 20}",
                "engine_type": "sglang",
                "model_name": "llama3.2",
                "max_tokens": 64,
            },
            headers=headers,
            timeout=30,
        )
        j = r.json()
        logger.info("POST /api/v1/llm/structured-generate -> code=%s data=%s", j.get("code"), j.get("data"))
    except Exception as e:
        logger.info("API structured-generate 请求失败: %s", e)

    logger.info("========== API 接口测试结束 ==========")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 推理：测试或启动 API 服务")
    parser.add_argument("--serve", action="store_true", help="启动 FastAPI 服务")
    parser.add_argument("--host", default="0.0.0.0", help="API 服务 host")
    parser.add_argument("--port", type=int, default=8000, help="API 服务 port")
    parser.add_argument("--api-test", action="store_true", help="运行 API 接口测试（需本机已起服务）")
    args = parser.parse_args()

    if args.serve:
        from api import FASTAPI_AVAILABLE, create_app
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("请安装 fastapi 与 uvicorn: pip install fastapi uvicorn")
        import uvicorn
        app = create_app()
        logger.info("启动 API 服务: %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.api_test:
        run_api_tests(f"http://127.0.0.1:{args.port}")
    else:
        run_core_tests()


if __name__ == "__main__":
    main()
