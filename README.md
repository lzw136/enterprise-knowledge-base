# 企业智能知识库问答系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

**基于 FastAPI + RAG + LLM 的企业级智能知识库问答系统**

[English](./README_EN.md) · [快速开始](#快速开始) · [项目架构](#项目架构) · [API文档](#api文档) · [部署指南](#docker部署)

</div>

---

## 📖 项目介绍

企业智能知识库问答系统是一个端到端的 RAG（检索增强生成）解决方案，支持：

- 📄 **多格式文档处理**：PDF、Word、TXT 等常见文档
- 🔍 **智能向量检索**：基于 ChromaDB 的语义搜索
- 🤖 **多模型支持**：DeepSeek / Qwen / OpenAI 等 OpenAI 兼容接口
- 💬 **多轮对话**：支持上下文理解的连续对话
- 🔄 **混合检索**：向量检索 + BM25 关键词检索融合
- 🛡️ **手写 ReAct Agent**：可追溯的推理决策过程

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI + Uvicorn | 高性能异步 API 框架 |
| **语言** | Python 3.10+ | 类型提示完整 |
| **向量数据库** | ChromaDB | 轻量级本地向量库 |
| **LLM** | DeepSeek / Qwen / OpenAI | OpenAI 兼容接口 |
| **Embedding** | text-embedding-3-small | OpenAI 官方嵌入 |
| **文档处理** | LangChain + Unstructured | 文档解析与分块 |
| **部署** | Docker + Docker Compose | 容器化部署 |

## ✨ 核心功能

```
┌─────────────────────────────────────────────────────────────┐
│                    用户查询流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户问题 ──► Query Rewriting ──► 混合检索                  │
│                                    │                        │
│                    ┌───────────────┼───────────────┐        │
│                    ▼               ▼               ▼        │
│              向量检索        BM25检索       知识图谱(规划)   │
│                    │               │               │        │
│                    └───────────────┼───────────────┘        │
│                                    ▼                        │
│                              检索融合                        │
│                                    │                        │
│                                    ▼                        │
│                              LLM 生成                       │
│                                    │                        │
│                                    ▼                        │
│                              最终答案                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/lzw136/enterprise-knowledge-base.git
cd enterprise-knowledge-base

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 3. 启动服务
docker-compose up -d

# 4. 访问服务
open http://localhost:8000/docs
```

### 方式二：本地开发

```bash
# 1. 克隆项目
git clone https://github.com/lzw136/enterprise-knowledge-base.git
cd enterprise-knowledge-base

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 5. 启动服务
uvicorn app.main:app --reload

# 6. 访问 API 文档
open http://localhost:8000/docs
```

## 📁 项目架构

```
enterprise-knowledge-base/
├── app/                        # 应用核心代码
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── api/                    # API 路由层
│   │   ├── chat.py            # 对话接口
│   │   ├── documents.py       # 文档管理接口
│   │   └── qa.py              # 问答接口
│   ├── core/                   # 核心模块
│   │   ├── llm.py             # LLM 调用封装
│   │   ├── embeddings.py      # 向量化模块
│   │   └── prompts.py         # Prompt 模板
│   ├── services/               # 业务服务层
│   │   ├── rag_service.py     # RAG 核心服务
│   │   ├── chat_service.py    # 对话服务
│   │   ├── document_service.py # 文档处理服务
│   │   └── agent_service.py   # ReAct Agent
│   ├── models/                 # 数据模型
│   │   ├── schemas.py         # Pydantic 模型
│   │   └── database.py        # 数据库模型
│   └── utils/                  # 工具函数
│       ├── text_splitter.py   # 文本分块
│       └── bm25.py            # BM25 检索
├── data/                       # 数据目录
│   ├── uploads/               # 上传文件
│   └── chroma_db/            # 向量数据库
├── tests/                      # 测试用例
├── scripts/                    # 脚本工具
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # 容器编排
├── requirements.txt           # Python 依赖
└── .env.example               # 环境变量示例
```

## 📡 API文档

### 健康检查

```bash
GET /health
```

### 上传文档

```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <file>
```

### 知识库问答

```bash
POST /api/v1/qa/ask
Content-Type: application/json

{
  "question": "公司的年假政策是什么？",
  "top_k": 5,
  "use_agent": false
}
```

### 多轮对话

```bash
POST /api/v1/chat/chat
Content-Type: application/json

{
  "message": "我想了解技术部门的加班制度",
  "session_id": "user-123-session-456",
  "use_rag": true
}
```

### 完整的 API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🐳 Docker部署

### 构建镜像

```bash
# 构建镜像
docker build -t knowledge-base:latest .

# 或者使用 docker-compose
docker-compose build
```

### 运行容器

```bash
# 使用 docker run
docker run -d \
  --name knowledge-base \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  knowledge-base:latest

# 使用 docker-compose（推荐）
docker-compose up -d
```

### 环境变量配置

在 `.env` 文件中配置：

```env
# LLM 配置
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=deepseek-chat

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small

# Chroma 配置
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# 应用配置
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### 生产环境建议

1. **使用反向代理**：Nginx + SSL
2. **配置健康检查**：利用 `/health` 端点
3. **数据备份**：定期备份 `data/chroma_db` 和 `data/uploads`
4. **资源限制**：根据 `docker-compose.yml` 中的配置调整

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行并显示详细输出
pytest -v

# 运行特定测试文件
pytest tests/test_rag_service.py

# 生成覆盖率报告
pytest --cov=app --cov-report=html
```

## 📚 学习资源

- [项目规划文档](../面试准备/AI知识库问答系统项目规划.md)
- [Week 1 学习指南](../面试准备/知识库项目_Week1学习指南.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件
