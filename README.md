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

### 2. 环境准备

- **Ollama**：安装并启动 [Ollama](https://ollama.com)，执行 `ollama pull llama3.2` 等拉取模型。
- **VLLM**：本地 `vllm serve <model>` 或配置远程 `base_url`。
- **SGLang**：本地启动 SGLang 服务（默认端口 30000）。

### 3. 运行方式

- **仅跑测试**（检测引擎可用性，Ollama 可用时会真实推理）：
  ```bash
  python llm_inference.py
  ```
- **启动 API 服务**（默认 `0.0.0.0:8000`）：
  ```bash
  python llm_inference.py --serve
  python llm_inference.py --serve --host 0.0.0.0 --port 8000
  ```
- **API 接口测试**（需先启动服务）：
  ```bash
  python llm_inference.py --api-test --port 8000
  ```

### 4. 接口文档

服务启动后：

- **Swagger UI**：http://localhost:8000/docs  
- **ReDoc**：http://localhost:8000/redoc  

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
