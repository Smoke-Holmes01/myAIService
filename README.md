# 中国古建筑 AI 小助手

这是一个面向比赛场景的本地部署项目，目标是把 `Qwen2.5-VL` 多模态模型和古建筑知识库结合起来，做成一个可以回答文字问题、理解古建筑图片的 AI 小助手。

## 项目能力

- 文本问答：回答中国古建筑相关问题。
- 图片理解：上传建筑图片后，结合视觉模型做识别与解读。
- RAG 检索增强：先从 PDF 知识库中检索相关资料，再交给大模型生成回答。
- 本地部署：通过 Flask 暴露 HTTP 接口，方便网页前端或比赛演示系统调用。

## 目录说明

- `qwen_server.py`：主服务入口。
- `build_kb.py`：把 PDF 资料构建为 Chroma 向量库。
- `rag_retriever.py`：RAG 检索封装。
- `test_kb.py`：测试知识库检索效果。
- `debug_load.py`：逐步检查模型加载是否成功。
- `qwen_server_debug.py`：调试模式启动服务。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

可以直接在服务器上设置环境变量，也可以参考 `.env.example`。

关键变量：

- `MODEL_PATH`：Qwen2.5-VL 模型目录
- `VECTOR_DB_PATH`：知识库目录
- `EMBEDDING_MODEL_NAME`：中文向量模型
- `EMBEDDING_DEVICE`：`cuda` 或 `cpu`
- `PORT`：服务端口

## 构建知识库

```bash
python build_kb.py --pdf-dir /home/yy/ancient_books --persist-dir /home/yy/my_ai_service/vector_db
```

如果显存紧张，也可以切到 CPU：

```bash
python build_kb.py --pdf-dir /home/yy/ancient_books --persist-dir /home/yy/my_ai_service/vector_db --device cpu
```

## 启动服务

```bash
python qwen_server.py
```

健康检查接口：

```bash
GET /api/ai/health
```

问答接口：

```bash
POST /api/ai/chat
Content-Type: application/json
```

纯文本问答请求示例：

```json
{
  "question": "斗拱的作用是什么？"
}
```

图片问答请求示例：

```json
{
  "question": "请帮我判断这张图里的建筑风格特点",
  "image": "data:image/jpeg;base64,..."
}
```

## 这次优化了什么

- 去掉了主程序、知识库、检索层中的硬编码路径。
- 支持通过环境变量切换模型路径、知识库路径和生成参数。
- 增加了服务启动入口和健康检查状态信息。
- 增加了请求校验，避免空问题或非法图片直接打崩服务。
- 修复了多模态回答解码不稳的问题，避免把提示词一起解出来。
- 知识库脚本改为命令行参数形式，更适合服务器部署和二次构建。

## 建议的下一步优化

- 增加对话历史记忆，让小助手连续对话更自然。
- 在响应中返回知识库来源片段，方便比赛答辩时解释“为什么这样回答”。
- 为前端增加流式输出接口，提升展示效果。
