# AGENTS.md

> 本文件面向 AI 编程助手。如果你对该项目一无所知，请先阅读本文件，再继续修改代码。

## 项目概览

本项目是**轻量 LLM 推理核心模块**，对标 Xinference 的基础推理能力，统一集成 **Ollama**、**vLLM**、**SGLang** 三大后端。采用分层解耦架构，提供 FastAPI 标准 HTTP API 与 Web 前端界面，便于大平台集成。

- **语言**：Python 3.10+（后端），原生 HTML/CSS/JS（前端）
- **主要自然语言**：中文（注释、文档、接口、前端 UI 均以中文为主）
- **无构建步骤**：纯 Python 项目，无需编译或打包
- **无正式测试框架**：测试逻辑直接写在 `main.py` 中

## 项目结构

```
main.py                 # 主入口：测试 / 启动 API 服务 / API 测试
config.json             # 运行时配置文件（含敏感 token，当前未被 .gitignore 排除）
config.example.json     # 配置示例（供用户复制为 config.json）
models.json             # 模型目录配置（前端展示的模型卡片、尺寸、引擎等）
models.example.json     # 模型目录示例
requirements.txt        # 依赖清单
log.txt                 # 运行时日志文件（文本）

frontend/               # 前端静态资源（纯原生 JS，无构建工具）
  index.html            # 单页应用：启动模型、运行模型、运行日志、推理抽屉
  css/style.css         # 深色主题样式（#1E88E5 主色）
  js/app.js             # 前端逻辑：模型卡片、参数配置、推理、日志拉取、运行列表

core/                   # 推理核心层
  __init__.py           # 对外导出 CONFIG、LLMInferencer、异常、模型下载等
  config.py             # 加载 config.json / models.json，初始化模型目录与环境变量
  exceptions.py         # 自定义异常类（LLMInferenceError 及其子类）
  inferencer.py         # LLMInferencer：统一推理入口，参数校验，引擎分发
  adapters/             # 引擎适配器
    base.py             # BaseLLMAdapter 抽象基类
    ollama.py           # OllamaAdapter（HTTP 调用本地 11434 端口，ollama 包为可选）
    vllm.py             # VLLMAdapter（支持本地加载或远程 API）
    sglang.py           # SGLangAdapter（仅远程 API，支持结构化输出）

services/               # 服务层
  instances.py          # 运行实例管理：start_model_impl、stop_model_impl、RUNNING_INSTANCES
  runtime_log.py        # 内存日志缓冲 + RuntimeLogHandler，供前端「运行日志」面板

api/                    # HTTP API 层（FastAPI）
  app.py                # create_app()：路由挂载、全局异常处理、CORS、请求日志、静态资源
  schemas.py            # Pydantic 请求/响应模型
  routes/
    models.py           # /api/v1/models：列表、启动、流式启动、运行中列表、停止
    llm.py              # /api/v1/llm：generate、chat、structured-generate
    logs.py             # /api/v1/logs：拉取近期运行日志

inference_core.py       # 兼容层：从 core 重新导出（旧代码可继续 from inference_core import ...）
llm_inference.py        # 兼容层：从 core/services/api 重新导出，__main__ 调用 main.main()
```

## 技术栈

### 必装依赖
```
pydantic>=2.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
requests>=2.31.0
httpx>=0.25.0
```

### 可选引擎依赖（按需安装）
- **Ollama**：`ollama` 包为可选；未安装时直接用 `httpx` 调本地 HTTP API
- **vLLM**：`vllm>=0.2.0`（需 GPU/CUDA）
- **SGLang**：`sglang[all]>=0.1.0`（需 GPU/CUDA）

### 模型下载（按需安装）
- `huggingface_hub>=0.20.0`（默认来源）
- `modelscope>=1.10.0`（国内来源，网络更稳定）

下载源由 `config.json` 中的 `model_source` 控制（`huggingface` 或 `modelscope`），在 `ensure_model_downloaded()` 中通过软导入使用。

## 运行与测试命令

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行核心推理测试（不启动服务，检测引擎可用性）
```bash
python main.py
```
测试逻辑在 `main.py` 的 `run_core_tests()` 中：检测各引擎可用性、执行单轮生成、多轮对话、结构化输出（如 SGLang 可用）。

### 启动 API 服务（默认 `0.0.0.0:8000`）
```bash
python main.py --serve
python main.py --serve --host 0.0.0.0 --port 8000
```
启动后访问：
- 前端界面：http://localhost:8000/
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc
- 健康检查：http://localhost:8000/health

### API 接口测试（需先启动服务）
```bash
python main.py --api-test --port 8000
```
测试逻辑在 `main.py` 的 `run_api_tests()` 中，依次请求 `/api/v1/models`、`/api/v1/models/running`、`/api/v1/llm/generate`、`/api/v1/llm/chat`、`/api/v1/llm/structured-generate`。

### 兼容旧入口
```bash
python llm_inference.py [--serve]
```

## 配置说明

### config.json（可选）
位于项目根目录，可通过环境变量 `CONFIG_PATH` 指定其他路径。不存在时使用内置默认配置。

关键配置项：
- `default_model_name`：默认模型名（如 `llama3.2`）
- `models_dir`：模型统一存储根目录（默认 `./models`）
- `models_subdir_ollama` / `models_subdir_hf`：Ollama 与 Hugging Face / ModelScope 缓存子目录
- `model_source`：模型下载来源，可选 `huggingface`（默认）或 `modelscope`
- `hf_token`：Hugging Face 令牌（`model_source` 为 `huggingface` 时生效）
- `ms_token`：ModelScope 令牌（`model_source` 为 `modelscope` 时生效）
- `ollama.base_url`：Ollama 服务地址，默认 `http://localhost:11434`
- `vllm.base_url`：vLLM 远程 API 地址；为 `null` 时本地加载
- `vllm.local_model_path`：本地 HF 格式模型目录
- `vllm.model_aliases`：友好名到 HF 模型 ID 或本地路径的映射
- `vllm.gpu_memory_utilization`：GPU 显存比例（0~1），默认 `0.65`
- `cors_origins`：CORS 允许来源列表，默认 `["*"]`；生产环境应设为具体域名数组
- `cors_allow_credentials`：CORS 是否允许携带凭证，默认 `false`
- `sglang.base_url`：SGLang 服务地址，默认 `http://localhost:30000`

程序启动时会根据 `models_dir` 创建目录，并自动设置环境变量 `OLLAMA_MODELS`、`HUGGINGFACE_HUB_CACHE` 与 `MODELSCOPE_CACHE`。

### models.json（可选）
用于配置前端展示的模型目录，每项包含 `id`、`name`、`description`、`official_url`、`sizes`（含 `hf_repo`、`ms_repo`、`ollama_name`）、`quantizations`、`engines`、`formats`。可通过环境变量 `MODELS_CONFIG` 指定路径。不存在时回退到内置的 `BUILTIN_MODELS`。

启动模型时，后端按 `model_id` + `size` 从 `models.json` 解析出 `hf_repo` / `ollama_name` 并执行下载或 Ollama 拉取。

## 代码组织与模块划分

### 1. 适配器模式（core/adapters/）
所有引擎适配器继承自 `BaseLLMAdapter`，必须实现：
- `engine_type`：引擎标识字符串
- `is_available()`：检查依赖是否已安装
- `check_service(model_name)`：检查服务是否就绪、模型是否存在
- `generate(...)`：单轮文本生成
- `chat(...)`：多轮对话
- `structured_generate(...)`：结构化输出（基类默认回退到 `generate`）

新增引擎：实现 `BaseLLMAdapter` 子类，并在 `LLMInferencer.ENGINE_MAP` 中注册即可，无需修改接口层。

### 2. 统一推理入口（core/inferencer.py）
`LLMInferencer` 根据 `engine_type` 实例化对应适配器，对外暴露：
- `generate(prompt, ...)`
- `chat(messages, ...)`
- `structured_generate(prompt, schema, ...)`（仅 SGLang）

构造时会检查适配器依赖是否安装（`is_available`）以及服务/模型是否可用（`check_service`）。参数校验包括：`temperature` 在 [0, 2]，`max_tokens` 为正整数，`top_p` 在 [0, 1]。

### 3. 运行实例管理（services/instances.py）
- `RUNNING_INSTANCES`：全局字典，`run_id` 到 `{inferencer, model_id, engine_type, ...}` 的映射
- `_running_lock`：`threading.Lock`，读写注册表时必须持有
- `start_model_impl(...)`：根据引擎类型启动模型
  - Ollama：尝试自动启动 `ollama serve`、检查/拉取模型、创建 Inferencer
  - vLLM/SGLang：先调用 `ensure_model_downloaded` 下载 HF 模型，再进程内加载
- `stop_model_impl(run_id)`：从注册表移除实例
- `get_running_inferencer(run_id)`：根据 `run_id` 返回已注册的 `LLMInferencer`
  - 支持 `ollama:` 前缀的 run_id（现场创建 Ollama Inferencer）

### 4. API 层（api/）
- 统一响应格式：`{ request_id, code, msg, data }`
- 全局异常处理：`LLMInferenceError` 子类映射为 400/500，`Exception` 映射为 500
- 请求中间件：为每个请求生成 `uuid` 作为 `request_id`，并记录访问日志
- CORS：`allow_origins=["*"]`（当前为全开放）

主要端点：
- `GET /api/v1/models`：列出模型目录
- `POST /api/v1/models/start`：启动模型（返回 run_id 与 address）
- `POST /api/v1/models/start-stream`：流式启动模型（返回 NDJSON 流）
- `GET /api/v1/models/running`：列出运行中的实例
- `POST /api/v1/models/running/{run_id}/stop`：停止指定实例
- `POST /api/v1/llm/generate`：单轮推理
- `POST /api/v1/llm/chat`：多轮对话
- `POST /api/v1/llm/structured-generate`：结构化输出（仅 SGLang）
- `GET /api/v1/logs`：拉取近期运行日志
- `POST /api/v1/test/proxy`：测试页代理，代为向模型引擎发送推理请求（解决浏览器跨域与 local:run_id 访问问题）

### 5. 前端（frontend/）
纯原生 JS，无构建工具。页面结构：
- 左侧参数配置面板（引擎、格式、大小、量化、GPU、副本等）
- 右侧主内容：启动模型（模型卡片）、运行模型（实例表格）、运行日志
- 推理抽屉：单轮生成 / 多轮对话

前端特性：
- 支持嵌入模式（URL 参数 `embed=1` 或在 iframe 中时隐藏侧栏）
- 模型列表从 `GET /api/v1/models` 动态加载，覆盖前端内置列表
- 运行列表从 `GET /api/v1/models/running` 拉取
- 日志从 `GET /api/v1/logs` 拉取，支持自动刷新（每 3 秒）

## 开发约定

### 编码风格
- 文件头统一使用 `# -*- coding: utf-8 -*-`
- 使用 `from __future__ import annotations` 支持 Python 3.10+ 类型注解
- 日志：通过 `logging.getLogger("模块名")` 获取 logger，避免直接 `print`
- 字符串格式化：代码中混用 f-string 与 `%` 格式化（以现有风格为准，不强制统一）
- 类型注解：鼓励使用，但未全量覆盖

### 异常处理
所有推理相关异常继承自 `LLMInferenceError`：
- `EngineNotInstalledError`：引擎依赖未安装
- `EngineNotRunningError`：引擎服务未启动或无法连接
- `ModelNotFoundError`：模型不存在或未拉取
- `InvalidParameterError`：参数不合法（temperature、max_tokens 等）
- `StructuredOutputNotSupportedError`：当前引擎不支持结构化输出

### 线程安全
- `RUNNING_INSTANCES` 和 `_log_buffer` 均使用 `threading.Lock` 保护
- vLLM 本地加载时会在进程内持有 `_llm` 实例（非线程安全，由 GIL 或单线程调用保证）

### 日志与运行日志的区别
- **Python logging**：输出到控制台/文件，供开发者排查问题
- **运行日志（runtime_log）**：内存缓冲（最多 500 条），通过 `GET /api/v1/logs` 返回给前端，记录启动/停止/失败等用户关心的关键事件。使用 `services.runtime_log.log_run_event(...)` 写入。

## 测试策略

- **无 pytest/unittest 等正式测试框架**。测试逻辑写在 `main.py` 的 `run_core_tests()` 和 `run_api_tests()` 中。
- `run_core_tests()`：检测各引擎可用性、执行单轮生成、多轮对话、结构化输出（如 SGLang 可用）。
- `run_api_tests()`：在已启动服务的前提下，依次请求各 API 端点并打印结果。
- 修改代码后建议：先 `python main.py` 跑核心测试，再 `--serve` 启动服务并用 `--api-test` 验证接口。

## 安全注意事项

1. **Token 安全**：`config.json` 中的 `hf_token`、`ms_token` 均为敏感信息。项目已配置 `.gitignore` 排除 `config.json` 与 `models.json`，但仍建议通过环境变量 `HF_TOKEN` / `MODELSCOPE_API_TOKEN` 注入令牌，避免在文件中留存真实 token。
2. **CORS**：`api/app.py` 中默认 `allow_origins=["*"]` 且 `allow_credentials=False`；可通过 `config.json` 的 `cors_origins` 与 `cors_allow_credentials` 调整。若部署到公网建议将 `cors_origins` 设为具体域名数组。
3. **无身份认证**：当前 API 无任何鉴权机制，生产环境部署需在外部网关或反向代理中补充。
4. **子进程调用**：`services/instances.py` 中会调用 `ollama serve` 和 `ollama pull` 子进程，需确保运行环境有对应命令且 PATH 正确。
5. **路径展开**：`vllm.local_model_path` 支持 `~` 展开为用户主目录，代码中通过 `os.path.expanduser` 处理。

## 常见问题排查

| 现象 | 排查方向 |
|------|----------|
| Ollama 服务未就绪 | 检查 `ollama serve` 是否已启动；本地环境会自动尝试启动 |
| vLLM 显存不足 | 调低 `config.json` 中的 `vllm.gpu_memory_utilization`（如 0.5）；停止其他 vLLM 实例 |
| 模型未找到 | Ollama 需先 `ollama pull <model>`；vLLM/SGLang 需正确配置 `hf_repo` 或 `local_model_path` |
| 结构化输出失败 | 仅 SGLang 支持，且需 SGLang 服务已启动 |
| 前端看不到后端日志 | Ollama 启动仅在前端 state 登记，不会调用 `POST /api/v1/models/start`；vLLM/SGLang 启动后端才有日志 |
