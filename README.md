# 轻量 LLM 推理核心模块

对标 Xinference 基础推理能力，统一集成 **Ollama**、**VLLM**、**SGLang** 三大后端，分层解耦，支持 FastAPI 标准 API，便于大平台集成。

## 快速开始

### 1. 依赖安装

**必装（核心 + API 层）：**
```bash
pip install -r requirements.txt
# 或
pip install pydantic fastapi uvicorn requests httpx
```

**各引擎可选：**
- **Ollama**：`pip install ollama`（不装则用内置 httpx 调本地 11434 端口）
- **VLLM**：`pip install vllm`（需 GPU/CUDA）
- **SGLang**：`pip install sglang`（需 GPU/CUDA）

### 2. 配置文件（可选）

项目根目录下的 **`config.json`** 用于统一配置默认模型名、各引擎地址和 vLLM 本地路径等，无需改代码。

- 若不存在 `config.json`，程序使用内置默认配置。
- 可通过环境变量 `CONFIG_PATH` 指定配置文件路径。
- 示例结构见 **`config.example.json`**，复制为 `config.json` 后按需修改。

**配置项说明：**

| 键 | 说明 |
|----|------|
| `default_model_name` | 默认模型名（如 `llama3.2`），创建推理实例时未传 `model_name` 时使用 |
| `models_dir` | 模型统一存储根目录（相对项目根或绝对路径），默认 `./models` |
| `models_subdir_ollama` | Ollama 模型子目录名，默认 `ollama`，实际路径为 `models_dir/ollama` |
| `models_subdir_hf` | Hugging Face 缓存子目录名，默认 `HF`，实际路径为 `models_dir/HF` |
| `hf_token` | Hugging Face 令牌（用于 gated/私有模型）。填写后程序会设置 `HF_TOKEN` 环境变量；不填则使用环境已有 `HF_TOKEN` 或 `huggingface-cli login` 的登录态。请勿提交含真实 token 的 config 到版本库 |
| `ollama.base_url` | Ollama 服务地址，默认 `http://localhost:11434` |
| `vllm.base_url` | vLLM 远程 API 地址；为 null 时使用本地加载 |
| `vllm.local_model_path` | vLLM **本地模型目录**（需为 **Hugging Face 格式**：含 `config.json`、tokenizer 等）。支持 `~` 表示家目录 |
| `vllm.model_aliases` | 友好名 → HF 模型 ID 或本地路径的映射；默认 `llama3.2` → `Qwen/Qwen2-0.5B-Instruct`（免授权）。若用 Meta Llama 等 gated 模型，需在 HF 申请并 `huggingface-cli login` 后改为 `meta-llama/Llama-3.2-3B-Instruct` 等 |
| `vllm.gpu_memory_utilization` | vLLM 使用的 GPU 显存比例（0~1），默认 `0.65`。若报错「Free memory ... is less than desired」可调低（如 0.5）或关闭其他占显存进程 |
| `sglang.base_url` | SGLang 服务地址，默认 `http://localhost:30000` |

**关于「模型统一存储目录」：**  
程序启动时会根据 `models_dir`、`models_subdir_ollama`、`models_subdir_hf` 在项目下创建目录，并设置 `OLLAMA_MODELS` 与 `HUGGINGFACE_HUB_CACHE`，使 Ollama 与 HF 下载的模型统一存到项目 `models/ollama`、`models/HF`（子目录名可在配置中修改）。若需 Ollama 使用该目录，请在本项目环境下启动 Ollama（或设置 `OLLAMA_MODELS` 后执行 `ollama pull`）；HF 模型在首次加载时会自动下载到 `models/HF`。

**关于「用 Ollama 拉取的模型路径」：**  
Ollama 拉取后的模型是其**自有格式**，不是 Hugging Face 的目录结构，**不能**直接作为 vLLM 的 `local_model_path`。若要用 vLLM 跑本地模型，需使用 **HF 格式** 的模型目录（可从 Hugging Face 下载或另存为 HF 格式），把该目录路径填到 `vllm.local_model_path` 或 `vllm.model_aliases` 中即可。

**模型目录（models.json）：**  
项目根目录下的 **`models.json`** 用于配置首页展示的**多模型**及每个模型可选的**大小、引擎、量化、格式**，前端参数配置会根据该文件动态显示选项，启动时按「模型 + 大小」解析出对应的 Hugging Face repo 或 Ollama 名称并下载/拉取。

- 若不存在 `models.json`，后端使用内置的少量模型列表（兼容旧行为）。
- 可通过环境变量 **`MODELS_CONFIG`** 指定模型配置路径。
- 示例结构见 **`models.example.json`**。每个模型需包含：
  - `id`、`name`、`description`、`official_url`
  - **`sizes`**：数组，每项 `{ "size": "0.5B", "hf_repo": "Qwen/Qwen2-0.5B-Instruct", "ollama_name": "qwen2:0.5b" }`
  - **`quantizations`**：如 `["none", "int4", "int8"]`
  - **`engines`**：如 `["ollama", "vllm", "sglang"]`
  - **`formats`**：如 `["pytorch", "safetensors"]`

启动模型时请在前端选择**模型 → 大小 → 引擎 → 量化/格式**，后端会根据 `model_id` + `size` 从 `models.json` 中解析出 `hf_repo` / `ollama_name` 并执行下载或 Ollama 拉取。

### 3. 环境准备

- **Ollama**：安装并启动 [Ollama](https://ollama.com)，执行 `ollama pull llama3.2` 等拉取模型。
- **VLLM**：本地 `vllm serve <model>` 或配置远程 `base_url`；或配置 `vllm.local_model_path` 指向 HF 格式的本地目录。
- **SGLang**：本地启动 SGLang 服务（默认端口 30000）。

### 4. 运行方式

**推荐使用主入口 `main.py`：**

- **仅跑测试**（检测引擎可用性，Ollama 可用时会真实推理）：
  ```bash
  python main.py
  ```
- **启动 API 服务**（默认 `0.0.0.0:8000`）：
  ```bash
  python main.py --serve
  python main.py --serve --host 0.0.0.0 --port 8000
  ```
  启动后访问 **http://localhost:8000/** 可打开前端界面（启动模型 / 运行模型，蓝白灰简洁风格）。
- **API 接口测试**（需先启动服务）：
  ```bash
  python main.py --api-test --port 8000
  ```

兼容旧入口：`python llm_inference.py [--serve]` 与上述等价。

### 5. Backend 代码结构

```
main.py                 # 主入口：python main.py [--serve] / 测试
config.json             # 配置文件（可选）
frontend/               # 前端静态资源

core/                   # 推理核心
  __init__.py           # 对外导出 CONFIG、LLMInferencer、异常等
  config.py             # 配置加载、模型目录、BUILTIN_MODELS、ensure_model_downloaded
  exceptions.py         # 异常类
  adapters/             # 引擎适配器
    base.py             # BaseLLMAdapter
    ollama.py / vllm.py / sglang.py
  inferencer.py         # LLMInferencer、validate_model_usable

services/               # 服务层
  instances.py          # 运行实例管理：start_model_impl、stop_model_impl、get_running_inferencer

api/                    # HTTP API 层
  app.py                # create_app()：FastAPI、中间件、异常处理、静态资源
  schemas.py            # Pydantic 请求/响应模型
  routes/
    models.py           # /api/v1/models、start、start-stream、running、stop
    llm.py              # /api/v1/llm/generate、chat、structured-generate

inference_core.py       # 兼容层：from core 重新导出（旧代码可继续 from inference_core import ...）
llm_inference.py       # 兼容层：from core/services/api 重新导出，__main__ 调用 main.main()
```

### 6. 接口文档

服务启动后：

- **前端界面**：http://localhost:8000/（启动模型、运行模型）
- **Swagger UI**：http://localhost:8000/docs  
- **ReDoc**：http://localhost:8000/redoc  

## 如何查看和测试所有后端接口

先启动服务：`python main.py --serve`（默认 http://0.0.0.0:8000）。

### 查看接口

| 方式 | 地址 | 说明 |
|------|------|------|
| **Swagger UI** | http://localhost:8000/docs | 可展开每个接口看请求体、响应，并直接「Try it out」发请求 |
| **ReDoc** | http://localhost:8000/redoc | 只读文档，排版适合阅读 |
| **健康检查** | http://localhost:8000/health | 返回 `{"status":"ok"}` 表示服务正常 |

### 后端接口清单

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/v1/models | 列出内置模型元信息（id、name、official_url、engines 等） |
| POST | /api/v1/models/start | 启动模型（body: `model_id`, `engine_type`，仅 vllm/sglang） |
| GET | /api/v1/models/running | 列出当前运行中的模型实例 |
| POST | /api/v1/models/running/{run_id}/stop | 停止指定 run_id 的实例 |
| POST | /api/v1/llm/generate | 单轮生成（body: `prompt`，以及 `run_id` 或 `engine_type`+`model_name`） |
| POST | /api/v1/llm/chat | 多轮对话（body: `messages`，以及 `run_id` 或 `engine_type`+`model_name`） |
| POST | /api/v1/llm/structured-generate | 结构化输出（仅 SGLang，body: `prompt`、可选 `schema` 等） |
| GET | /health | 健康检查 |
| GET | / | 前端首页 |

### 用 curl 快速测试（需先启动服务）

```bash
# 1. 健康检查
curl -s http://localhost:8000/health

# 2. 列出内置模型
curl -s http://localhost:8000/api/v1/models

# 3. 列出运行中的模型
curl -s http://localhost:8000/api/v1/models/running

# 4. 启动模型（vllm，会下载并加载，耗时较长）
curl -s -X POST http://localhost:8000/api/v1/models/start \
  -H "Content-Type: application/json" \
  -d '{"model_id":"qwen2-0.5b","engine_type":"vllm"}'

# 5. 停止已启动的模型（将上一步返回的 run_id 替换到下面）
# curl -s -X POST http://localhost:8000/api/v1/models/running/<run_id>/stop

# 6. 单轮生成（不依赖已启动实例，直接指定引擎和模型）
curl -s -X POST http://localhost:8000/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","engine_type":"ollama","model_name":"llama3.2","max_tokens":32}'

# 7. 多轮对话
curl -s -X POST http://localhost:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"1+1=?"}],"engine_type":"ollama","model_name":"llama3.2","max_tokens":32}'

# 8. 使用已启动实例做生成（用 run_id 替代 engine_type/model_name）
# curl -s -X POST http://localhost:8000/api/v1/llm/generate \
#   -H "Content-Type: application/json" \
#   -d '{"prompt":"你好","run_id":"<run_id>","max_tokens":32}'
```

### 用脚本跑一轮 API 测试

在**已启动服务**的前提下，在项目目录执行：

```bash
python llm_inference.py --api-test --port 8000
```

会依次请求：`/api/v1/models`、`/api/v1/models/running`、`/api/v1/llm/generate`、`/api/v1/llm/chat`、`/api/v1/llm/structured-generate`，并在终端打印每个接口的 `code` 与 `data`。其中 generate/chat 使用 `engine_type=ollama`，需本机已起 Ollama 且已拉取对应模型；structured-generate 使用 sglang，若未起 SGLang 会返回错误，属正常。

## 使用示例

### 原生 Python 调用

```python
from llm_inference import LLMInferencer

# 初始化（Ollama）
inferencer = LLMInferencer(engine_type="ollama", model_name="llama3.2")

# 单轮推理
text = inferencer.generate("介绍一下北京。", max_tokens=256)

# 多轮对话（OpenAI messages 格式）
messages = [
    {"role": "user", "content": "我叫张三"},
    {"role": "assistant", "content": "你好张三！"},
    {"role": "user", "content": "我叫什么？"},
]
reply = inferencer.chat(messages, max_tokens=128)

# 切换引擎
inferencer_vllm = LLMInferencer(engine_type="vllm", model_name="Qwen/Qwen-7B")
inferencer_sglang = LLMInferencer(engine_type="sglang", model_name="llama3.2")

# 结构化输出（仅 SGLang）
result = inferencer_sglang.structured_generate(
    "请返回 {\"city\": \"北京\", \"population\": 数字}",
    schema={"city": "string", "population": "number"},
)
```

### API 调用示例

```bash
# 单轮推理
curl -X POST http://localhost:8000/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","engine_type":"ollama","model_name":"llama3.2","max_tokens":32}'

# 多轮对话
curl -X POST http://localhost:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"1+1=?"}],"engine_type":"ollama","model_name":"llama3.2","max_tokens":32}'

# 结构化输出（仅 SGLang）
curl -X POST http://localhost:8000/api/v1/llm/structured-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"输出 JSON: {\"name\":\"张三\",\"age\":20}","engine_type":"sglang","model_name":"llama3.2"}'
```

### 统一响应格式

- **成功**：`{"request_id":"uuid","code":200,"msg":"success","data":{...}}`
- **失败**：`{"request_id":"uuid","code":400|500,"msg":"错误原因","data":null}`

## 前端与后端对接说明

为避免「在前端操作却看不到后端日志」的困惑，下面明确列出**哪些前端操作会请求后端**、哪些不会，以及**尚未打通的点**。

### 会触发后端请求的操作

| 前端操作 | 后端接口 | 说明 |
|----------|----------|------|
| 打开页面 / 加载模型列表 | `GET /api/v1/models` | 用于「启动模型」页的模型卡片（当前为内置的 qwen2-0.5b、llama3.2） |
| 选择 **vLLM** 或 **SGLang** 后点击「启动」 | `POST /api/v1/models/start` | 会下载模型（若未下载）、在进程内加载并登记到运行列表；**此时后端会有日志** |
| 进入「运行模型」页 | `GET /api/v1/models/running` | 拉取当前由后端启动的实例列表 |
| 在运行列表中点击「停止」 | `POST /api/v1/models/running/{run_id}/stop` | 仅对通过「启动」接口启动的实例有效（有 `run_id`） |
| 在运行列表点击「推理」→ 单轮「生成」 | `POST /api/v1/llm/generate` | 可传 `run_id`（用已启动实例）或 `engine_type`+`model_name` |
| 在运行列表点击「推理」→ 多轮「发送」 | `POST /api/v1/llm/chat` | 同上 |

### 不会触发后端请求的操作（设计如此）

| 前端操作 | 原因 |
|----------|------|
| 选择 **Ollama** 后点击「启动」 | Ollama 由用户本地单独启动，本平台不负责拉取/加载；前端仅把该模型**记入本地运行列表**用于展示和推理。因此**不会**调用 `POST /api/v1/models/start`，后端无日志属正常。 |
| 点击模型卡片、打开/关闭配置面板、切换 tab | 纯前端交互，无 API 调用。 |

### 当前设计下的限制与可扩展点

1. **启动接口仅支持 vllm/sglang**  
   `POST /api/v1/models/start` 的请求体里 `engine_type` 只接受 `vllm` 或 `sglang`。Ollama 不通过该接口「启动」，只通过推理接口（`/generate`、`/chat`）用 `engine_type=ollama` 直接调本地 Ollama 服务。

2. **配置面板参数未全部传给后端**  
   前端配置里的「模型格式、模型大小、量化、GPU 数量、副本、思考模式」等目前**没有**传给 `POST /api/v1/models/start`（该接口只收 `model_id`、`engine_type`）。若需要按这些参数影响加载或调度，需在后端扩展接口并在前端组请求体。

3. **嵌入模型**  
   前端「嵌入模型」tab 下的卡片是写死的静态列表，**没有**对应的 `GET /api/v1/models` 嵌入模型列表或启动接口；若要统一管理，需后端增加嵌入模型元数据与启动/运行接口。

4. **结构化输出**  
   后端已有 `POST /api/v1/llm/structured-generate`（仅 SGLang），前端推理抽屉里**没有**「结构化输出」入口，需要可再加一个 tab 或按钮并传 `schema` 等参数。

5. **运行列表同步**  
   通过「启动」登记的 Ollama 条目只在前端 state 中，刷新页面或再次请求 `GET /api/v1/models/running` 时不会出现；只有 vllm/sglang 的实例会从后端返回并显示。

若你希望「选 Ollama 点启动」时也在后端留痕（例如只做登记或健康检查），可以增加一个轻量接口（如 `POST /api/v1/models/register`）由前端在选 Ollama 时调用，并在 README 中注明用途。

## 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 服务未启动 | Ollama/VLLM/SGLang 未运行 | 先启动对应服务，Ollama 可直接运行 `ollama serve` |
| 显存不足 | VLLM/SGLang 需 GPU | 换小模型或使用 Ollama 等已做显存优化的方案 |
| 库未安装 | 未装 vllm/sglang/ollama | 按 `requirements.txt` 安装对应引擎可选依赖 |
| 模型不存在 | 模型未拉取或路径错误 | Ollama 用 `ollama list` 查看并 `ollama pull <model>` |
| 参数错误 | temperature/max_tokens 等不合法 | 使用 0–2 的 temperature、正整数 max_tokens，接口会返回 400 及具体原因 |
| 结构化输出仅 SGLang | 其他引擎未实现 | 使用 `engine_type=sglang` 或改用普通 generate/chat |

## 项目结构（单文件组织）

- 依赖声明 → 库导入（软导入）→ 日志配置 → 自定义异常  
- 引擎适配基类 `BaseLLMAdapter` → Ollama/VLLM/SGLang 适配类  
- 核心推理类 `LLMInferencer`  
- FastAPI 接口层（Pydantic 校验、统一响应、全局异常、CORS）  
- 测试用例与启动入口  

扩展新引擎：实现 `BaseLLMAdapter` 子类并在 `LLMInferencer.ENGINE_MAP` 中注册即可，无需改接口层。
