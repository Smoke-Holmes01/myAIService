# 中国古建筑 AI 小助手

这是一个面向比赛展示和本地部署场景的 AI 项目，核心目标是将 `Qwen2.5-VL` 多模态大模型与中国古建筑领域知识结合，构建一个可以进行文字问答、图片识别与风格分析的古建筑 AI 小助手。

项目当前支持两种运行模式：

- 纯模型模式：直接使用 `Qwen2.5-VL` 进行文本和图片问答。
- RAG 增强模式：在纯模型基础上增加古建筑 PDF 知识库检索，提高回答的专业性和可解释性。

## 1. 项目功能

- 文本问答：回答中国古建筑相关问题。
- 图片识别：识别上传图片中的古建筑，并分析建筑风格特征。
- HTTP 接口服务：使用 Flask 提供后端接口，方便网页、比赛前端或其他系统调用。
- 可选知识库增强：通过 PDF 构建 Chroma 向量库，实现 RAG 检索增强回答。

## 2. 目录结构

- `qwen_server.py`
  主服务入口，加载模型并提供 `/api/ai/health` 和 `/api/ai/chat` 接口。

- `qwen_server_debug.py`
  调试模式启动服务，便于开发排查。

- `debug_load.py`
  单独测试模型能否正确加载，适合部署前验证环境。

- `build_kb.py`
  读取 PDF 资料，切分文本并构建 Chroma 向量知识库。

- `rag_retriever.py`
  RAG 检索封装，负责加载知识库并检索上下文。

- `test_kb.py`
  对知识库进行简单检索测试。

- `requirements.txt`
  项目依赖清单。

- `.env.example`
  环境变量示例文件。

## 3. 运行环境建议

推荐环境：

- Linux 服务器
- Python 3.10 到 3.12
- NVIDIA GPU
- CUDA 可用
- 已提前下载好 `Qwen2.5-VL` 模型

如果只做演示，纯模型模式即可运行；如果想提高专业度和答辩说服力，建议额外构建知识库。

## 4. 安装依赖

进入项目目录后安装依赖：

```bash
pip install -r requirements.txt
```

如果后续遇到 `langchain.text_splitter` 相关兼容问题，可补装：

```bash
pip install langchain-text-splitters
```

## 5. 环境变量说明

项目主要通过环境变量控制模型路径、知识库路径和生成参数。可以直接在终端 `export`，也可以参考 `.env.example`。

常用变量如下：

- `MODEL_PATH`
  大模型目录，默认值：
  `/home/yy/models/qwen2.5-vl-72b`

- `HOST`
  服务监听地址，默认值：
  `0.0.0.0`

- `PORT`
  服务端口，默认值：
  `11435`

- `ENABLE_RAG`
  是否启用知识库检索，默认值：
  `true`

- `VECTOR_DB_PATH`
  向量知识库目录，默认值：
  `/home/yy/my_ai_service/vector_db`

- `EMBEDDING_MODEL_NAME`
  中文向量模型名称，默认值：
  `shibing624/text2vec-base-chinese`

- `EMBEDDING_DEVICE`
  向量模型运行设备，可选：
  `cuda` 或 `cpu`

- `RAG_TOP_K`
  检索返回的片段数，默认值：
  `3`

- `MAX_NEW_TOKENS`
  大模型最大生成长度，默认值：
  `512`

- `TEMPERATURE`
  生成温度，默认值：
  `0.7`

- `TOP_P`
  采样参数，默认值：
  `0.9`

## 6. 启动前检查

正式启动服务前，建议先检查模型是否能正确加载：

```bash
python debug_load.py
```

如果输出中出现类似内容，说明模型环境正常：

```text
模型加载成功，输入设备=cuda:0
```

如果报错 `accelerate` 缺失，安装：

```bash
pip install accelerate
```

## 7. 启动服务

### 7.1 纯模型模式启动

如果暂时不使用知识库，建议显式关闭 RAG：

```bash
export ENABLE_RAG=false
python qwen_server.py
```

### 7.2 RAG 模式启动

如果已经构建了知识库：

```bash
export ENABLE_RAG=true
export VECTOR_DB_PATH=/home/yy/my_ai_service/vector_db
python qwen_server.py
```

启动成功后通常会看到：

```text
服务启动成功，监听 0.0.0.0:11435
Running on http://127.0.0.1:11435
```

说明服务已经可以接受请求。

## 8. 接口说明

### 8.1 健康检查接口

请求：

```bash
curl http://127.0.0.1:11435/api/ai/health
```

示例返回：

```json
{
  "device": "cuda:0",
  "model_loaded": true,
  "model_path": "/home/yy/models/qwen2.5-vl-72b",
  "rag_loaded": false,
  "service": "Chinese Ancient Architecture AI Assistant",
  "status": "up"
}
```

字段说明：

- `model_loaded`
  是否成功加载大模型。

- `rag_loaded`
  是否成功加载知识库。

- `status`
  服务状态。

### 8.2 文本问答接口

请求示例：

```bash
curl -X POST "http://127.0.0.1:11435/api/ai/chat" \
  -H "Content-Type: application/json" \
  --data-raw '{"question":"请介绍一下斗拱的作用"}'
```

返回示例：

```json
{
  "question": "请介绍一下斗拱的作用",
  "answer": "斗拱是中国古代木构建筑中的重要构件，主要用于承托屋檐、分散荷载，并具有装饰作用。",
  "used_knowledge": false,
  "has_image": false
}
```

### 8.3 图片识别接口

方式一，使用 `curl`：

```bash
IMG_B64=$(base64 -w 0 "/home/yy/my_ai_service/故宫.jpg")
curl -X POST "http://127.0.0.1:11435/api/ai/chat" \
  -H "Content-Type: application/json" \
  --data-raw "{\"question\":\"请识别这张图片中的建筑，并简要说明它的风格特点\",\"image\":\"data:image/jpeg;base64,${IMG_B64}\"}"
```

方式二，使用 Python 发送请求，通常更稳定：

```bash
python - <<'PY'
import base64
import json
import urllib.request

img_path = "/home/yy/my_ai_service/故宫.jpg"
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "question": "请识别这张图片中的建筑，并简要说明它的风格特点",
    "image": f"data:image/jpeg;base64,{img_b64}",
}

data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:11435/api/ai/chat",
    data=data,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    print(resp.read().decode("utf-8"))
PY
```

返回中如果 `has_image` 为 `true`，说明图片已被正确送入模型。

## 9. 构建知识库

如果服务器上已经准备好古建筑 PDF，可以构建向量知识库：

```bash
python build_kb.py --pdf-dir /home/yy/ancient_books --persist-dir /home/yy/my_ai_service/vector_db
```

如果想避免占用 GPU，也可以改用 CPU：

```bash
python build_kb.py --pdf-dir /home/yy/ancient_books --persist-dir /home/yy/my_ai_service/vector_db --device cpu
```

构建完成后可检查目录：

```bash
ls /home/yy/my_ai_service/vector_db
```

然后测试知识库检索：

```bash
python test_kb.py --vector-db-path /home/yy/my_ai_service/vector_db
```

## 10. 部署建议流程

建议按照以下顺序部署：

1. 拉取代码

```bash
git pull origin test01
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 验证模型加载

```bash
python debug_load.py
```

4. 启动纯模型服务进行接口验证

```bash
export ENABLE_RAG=false
python qwen_server.py
```

5. 如果纯模型模式正常，再考虑构建知识库并启用 RAG

## 11. 常见问题排查

### 11.1 `accelerate` 缺失

报错特征：

```text
Using `device_map` requires `accelerate`
```

解决方法：

```bash
pip install accelerate
```

### 11.2 知识库目录不存在

报错特征：

```text
知识库目录不存在: /home/yy/my_ai_service/vector_db
```

说明知识库还未构建，或者路径配置错误。

解决方法：

- 不使用知识库时：

```bash
export ENABLE_RAG=false
python qwen_server.py
```

- 使用知识库时，先构建：

```bash
python build_kb.py --pdf-dir /home/yy/ancient_books --persist-dir /home/yy/my_ai_service/vector_db
```

### 11.3 PDF 文件损坏

报错特征：

```text
invalid pdf header
EOF marker not found
Stream has ended unexpectedly
```

说明某个 PDF 文件损坏或格式异常。

解决方法：

- 临时移走损坏 PDF 再构建知识库
- 或修改 `build_kb.py`，对损坏 PDF 做跳过处理

### 11.4 `langchain.text_splitter` 导入失败

报错特征：

```text
No module named 'langchain.text_splitter'
```

解决方法：

```bash
pip install langchain-text-splitters
```

并将导入改为：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### 11.5 图片或问答返回较慢

这是正常现象，原因通常是：

- 使用的是 72B 多模态模型
- 图片输入比纯文本推理更慢
- GPU 正在进行较长生成

建议：

- 先等待 1 到 3 分钟
- 适当降低 `MAX_NEW_TOKENS`
- 比赛展示时优先准备较清晰、结构明显的建筑图片

## 12. 当前项目状态

当前版本已经完成：

- 纯模型模式启动验证
- 健康检查接口验证
- 文本问答接口验证
- 图片识别接口验证

当前仍建议后续完善的方向：

- 为知识库构建加入坏 PDF 自动跳过逻辑
- 增加多轮对话记忆
- 增加知识来源返回，提升答辩可解释性
- 增加流式输出，改善展示效果

## 13. 安全说明

- 不要把 GitHub token、服务器密码、模型私有地址直接写进仓库。
- 如果误把 token 发到聊天、截图或代码中，应立即去 GitHub 撤销并重新生成。
- 建议后续改用 GitHub SSH key 推送代码，减少重复输入 token 的风险。
