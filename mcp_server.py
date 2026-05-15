"""
MCP Server - 企业智能知识库
将知识库暴露为 MCP 工具，供 Claude Desktop / Cursor / VS Code 等客户端调用

支持两种运行模式：
  1. 本地模式（默认）：直接调用服务层，低延迟，适合单机部署
     python mcp_server.py
  2. HTTP 模式：通过 API 调用，松耦合，适合服务端部署
     python mcp_server.py http
  3. SSE 模式：HTTP Server-Sent Events 传输
     python mcp_server.py sse

配置：通过 .env 文件管理 API Key 等敏感信息，不要硬编码
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ==================== 环境变量加载 ====================
from dotenv import load_dotenv
load_dotenv()

# ==================== 运行模式 ====================
# local: 直接调用服务层（默认，低延迟）
# http: 通过 FastAPI HTTP API 调用（松耦合，支持远程部署）
MCP_MODE = os.getenv("MCP_MODE", "local")
API_BASE_URL = os.getenv("MCP_API_BASE_URL", "http://localhost:8000")

# 初始化 MCP 服务器
mcp = FastMCP(
    "enterprise-knowledge-base",
    description="企业智能知识库 - 支持文档管理、语义搜索、智能问答",
)

# ==================== HTTP 模式工具函数 ====================

def _http_get(path: str) -> dict:
    """HTTP GET 请求（用于 HTTP 模式）"""
    import requests
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": f"无法连接知识库服务: {API_BASE_URL}，请确认服务已启动"}
    except requests.Timeout:
        return {"error": "请求超时，知识库服务响应过慢"}
    except Exception as e:
        return {"error": str(e)}


def _http_post(path: str, json_data: dict = None, files: dict = None) -> dict:
    """HTTP POST 请求（用于 HTTP 模式）"""
    import requests
    try:
        resp = requests.post(
            f"{API_BASE_URL}{path}",
            json=json_data,
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": f"无法连接知识库服务: {API_BASE_URL}，请确认服务已启动"}
    except requests.Timeout:
        return {"error": "请求超时，知识库服务响应过慢"}
    except Exception as e:
        return {"error": str(e)}


def _http_delete(path: str) -> dict:
    """HTTP DELETE 请求（用于 HTTP 模式）"""
    import requests
    try:
        resp = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": f"无法连接知识库服务: {API_BASE_URL}，请确认服务已启动"}
    except Exception as e:
        return {"error": str(e)}


# ==================== 本地模式服务层初始化（延迟加载） ====================

_rag_service = None
_document_service = None


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from app.services.rag_service import RAGService
        _rag_service = RAGService()
    return _rag_service


def _get_document_service():
    global _document_service
    if _document_service is None:
        from app.services.document_service import DocumentService
        _document_service = DocumentService()
    return _document_service


# ==================== MCP 工具 ====================


@mcp.tool()
def health_check() -> str:
    """检查知识库系统运行状态。返回系统健康信息和配置情况。"""
    if MCP_MODE == "http":
        result = _http_get("/health")
        if "error" in result:
            return f"❌ 服务不可用: {result['error']}"
        status = _http_get("/api/v1/status")
        return (
            f"✅ 系统状态: {result.get('status', '未知')}\n"
            f"版本: {result.get('version', '未知')}\n"
            f"LLM 已配置: {status.get('llm_configured', False)}\n"
            f"模型: {status.get('model', '未知')}"
        )
    else:
        try:
            from app.config import settings
            return (
                f"✅ 系统状态: healthy\n"
                f"运行模式: local（直接调用服务层）\n"
                f"LLM 已配置: {bool(settings.openai_api_key)}\n"
                f"模型: {settings.openai_model}"
            )
        except Exception as e:
            return f"❌ 服务初始化失败: {str(e)}"


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在企业知识库中搜索相关信息。返回匹配的文档片段及相关度分数。

    Args:
        query: 搜索查询内容
        top_k: 返回结果数量，默认 5
    """
    if MCP_MODE == "http":
        # HTTP 模式没有独立的 retrieve 端点，用 qa 接口获取 sources
        result = _http_post("/api/v1/qa/", json_data={
            "question": query,
            "top_k": top_k,
            "include_sources": True,
        })
        if "error" in result:
            return f"搜索失败: {result['error']}"
        sources = result.get("sources", [])
        if not sources:
            return "未找到相关信息"
        formatted = []
        for i, src in enumerate(sources, 1):
            name = src.get("document_name", "未知文件")
            score = src.get("score", 0)
            content = src.get("content", "")[:300]
            formatted.append(f"[{i}] 文件: {name} | 相关度: {score:.2f}\n{content}\n")
        return "\n".join(formatted)

    # 本地模式
    try:
        rag = _get_rag_service()
        results = rag.retrieve(query, top_k=top_k)
        if not results:
            return "未找到相关信息"

        formatted = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:300]
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "未知文件")
            formatted.append(
                f"[{i}] 文件: {filename} | 相关度: {score:.2f}\n{content}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"搜索失败: {str(e)}"


@mcp.tool()
def ask_question(question: str, top_k: int = 5) -> str:
    """基于企业知识库回答问题。会检索相关文档并生成综合回答。

    Args:
        question: 要回答的问题
        top_k: 检索的文档块数量，默认 5
    """
    if MCP_MODE == "http":
        result = _http_post("/api/v1/qa/", json_data={
            "question": question,
            "top_k": top_k,
            "include_sources": True,
        })
        if "error" in result:
            return f"问答失败: {result['error']}"
        answer = result.get("answer", "无法生成回答")
        sources = result.get("sources", [])
        output = f"## 回答\n{answer}\n"
        if sources:
            output += "\n## 参考来源\n"
            for i, src in enumerate(sources, 1):
                name = src.get("document_name", "未知文件")
                score = src.get("score", 0)
                content = src.get("content", "")[:150]
                output += f"[{i}] {name} (相关度: {score:.2f}): {content}...\n"
        return output

    # 本地模式
    try:
        rag = _get_rag_service()
        result = rag.ask(question, top_k=top_k, include_sources=True)

        answer = result.get("answer", "无法生成回答")
        sources = result.get("sources", [])

        output = f"## 回答\n{answer}\n"
        if sources:
            output += "\n## 参考来源\n"
            for i, src in enumerate(sources, 1):
                name = src.document_name or "未知文件"
                score = src.score
                content = src.content[:150]
                output += f"[{i}] {name} (相关度: {score:.2f}): {content}...\n"

        return output
    except Exception as e:
        return f"问答失败: {str(e)}"


@mcp.tool()
def upload_document(file_path: str) -> str:
    """上传本地文件到知识库。支持 .txt, .md, .pdf, .docx 格式。

    注意：file_path 是 MCP Server 运行机器上的绝对路径。
    在远程部署场景下，请使用 HTTP API 的上传接口。

    Args:
        file_path: MCP Server 运行机器上的文件绝对路径
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"文件不存在: {file_path}。请确认路径是 MCP Server 运行机器上的绝对路径。"

        if MCP_MODE == "http":
            with open(path, "rb") as f:
                result = _http_post(
                    "/api/v1/documents/",
                    files={"file": (path.name, f)},
                )
            if "error" in result:
                return f"上传失败: {result['error']}"
            return (
                f"上传成功!\n"
                f"文档ID: {result.get('id', '未知')}\n"
                f"文件名: {path.name}"
            )

        # 本地模式
        doc_service = _get_document_service()
        rag = _get_rag_service()

        # 解析文档
        text = doc_service.parse_document(str(path))
        if not text:
            return f"文档解析失败或内容为空: {path.name}"

        # 分块
        chunks = doc_service.split_documents(text)
        if not chunks:
            return f"文档分块后无内容: {path.name}"

        # 生成文档 ID 并存储
        document_id = str(uuid.uuid4())
        metadata = {
            "id": document_id,
            "filename": path.name,
            "file_size": path.stat().st_size,
            "file_type": path.suffix.lower().strip("."),
            "uploaded_at": datetime.now().isoformat(),
            "chunk_count": len(chunks),
        }

        doc_service.store_document(document_id, chunks, metadata)
        doc_service._metadata[document_id] = metadata
        doc_service._save_metadata()

        return (
            f"上传成功!\n"
            f"文档ID: {document_id}\n"
            f"文件名: {path.name}\n"
            f"分块数: {len(chunks)}"
        )
    except Exception as e:
        return f"上传失败: {str(e)}"


@mcp.tool()
def list_documents() -> str:
    """列出知识库中所有已上传的文档。"""
    if MCP_MODE == "http":
        result = _http_get("/api/v1/documents/")
        if "error" in result:
            return f"获取文档列表失败: {result['error']}"
        docs = result.get("documents", [])
        if not docs:
            return "知识库中暂无文档"
        formatted = [f"共 {len(docs)} 份文档:\n"]
        for doc in docs:
            doc_id = doc.get("id", doc.get("document_id", "未知"))
            filename = doc.get("filename", "未知")
            file_type = doc.get("file_type", "未知")
            chunk_count = doc.get("chunk_count", 0)
            formatted.append(
                f"- {filename} | ID: {doc_id} | 类型: {file_type} | 块数: {chunk_count}"
            )
        return "\n".join(formatted)

    # 本地模式
    try:
        doc_service = _get_document_service()
        docs = doc_service.list_documents()

        if not docs:
            return "知识库中暂无文档"

        formatted = [f"共 {len(docs)} 份文档:\n"]
        for doc in docs:
            doc_id = doc.get("id", "未知")
            filename = doc.get("filename", "未知")
            file_type = doc.get("file_type", "未知")
            chunk_count = doc.get("chunk_count", 0)
            uploaded_at = doc.get("uploaded_at", "未知")
            formatted.append(
                f"- {filename} | ID: {doc_id} | 类型: {file_type} | "
                f"块数: {chunk_count} | 上传: {uploaded_at}"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"获取文档列表失败: {str(e)}"


@mcp.tool()
def get_document(document_id: str) -> str:
    """获取指定文档的详细信息。

    Args:
        document_id: 文档 ID
    """
    if MCP_MODE == "http":
        result = _http_get(f"/api/v1/documents/{document_id}")
        if "error" in result:
            return f"获取文档信息失败: {result['error']}"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    # 本地模式
    try:
        doc_service = _get_document_service()
        doc = doc_service.get_document(document_id)

        if not doc:
            return f"未找到文档: {document_id}"

        return json.dumps(doc, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"获取文档信息失败: {str(e)}"


@mcp.tool()
def delete_document(document_id: str) -> str:
    """从知识库中删除指定文档。

    Args:
        document_id: 要删除的文档 ID
    """
    if MCP_MODE == "http":
        result = _http_delete(f"/api/v1/documents/{document_id}")
        if "error" in result:
            return f"删除文档失败: {result['error']}"
        return f"已删除文档 (ID: {document_id})"

    # 本地模式
    try:
        doc_service = _get_document_service()
        doc = doc_service.get_document(document_id)

        if not doc:
            return f"未找到文档: {document_id}"

        filename = doc.get("filename", "未知")
        success = doc_service.delete_document(document_id)

        if success:
            return f"已删除文档: {filename} (ID: {document_id})"
        else:
            return f"删除失败: {document_id}"
    except Exception as e:
        return f"删除文档失败: {str(e)}"


@mcp.tool()
def get_knowledge_base_stats() -> str:
    """获取知识库统计信息，包括文档数量、块数、向量库记录数等。"""
    if MCP_MODE == "http":
        result = _http_get("/api/v1/documents/stats")
        if "error" in result:
            # 降级：用 status 接口
            status = _http_get("/api/v1/status")
            return (
                f"系统状态: {status.get('status', '未知')}\n"
                f"LLM 已配置: {status.get('llm_configured', False)}\n"
                f"模型: {status.get('model', '未知')}"
            )
        return (
            f"文档总数: {result.get('document_count', 0)}\n"
            f"文档块总数: {result.get('total_chunks', 0)}\n"
            f"ChromaDB 记录数: {result.get('collection_count', 0)}"
        )

    # 本地模式
    try:
        doc_service = _get_document_service()
        rag = _get_rag_service()
        docs = doc_service.list_documents()
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        return (
            f"文档总数: {len(docs)}\n"
            f"文档块总数: {total_chunks}\n"
            f"ChromaDB 记录数: {rag.collection.count()}"
        )
    except Exception as e:
        return f"获取统计信息失败: {str(e)}"


# ==================== MCP 资源 ====================


@mcp.resource("knowledge://documents")
def get_documents_resource() -> str:
    """知识库文档列表资源"""
    if MCP_MODE == "http":
        result = _http_get("/api/v1/documents/")
        if "error" in result:
            return f"获取文档列表失败: {result['error']}"
        return json.dumps(result.get("documents", []), ensure_ascii=False, indent=2, default=str)

    try:
        doc_service = _get_document_service()
        docs = doc_service.list_documents()

        if not docs:
            return "知识库中暂无文档"

        return json.dumps(docs, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"获取文档列表失败: {str(e)}"


# ==================== 入口 ====================

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport == "http":
        # HTTP 模式：通过 API 调用，需要先启动 FastAPI 服务
        os.environ["MCP_MODE"] = "http"
        print(f"🌐 MCP Server 运行在 HTTP 模式，API 地址: {API_BASE_URL}", file=sys.stderr)
        mcp.run(transport="stdio")
    elif transport == "sse":
        from app.config import settings
        mcp.run(
            transport="sse",
            host=settings.mcp_host,
            port=settings.mcp_port,
        )
    else:
        # 默认本地模式：直接调用服务层
        print("🏠 MCP Server 运行在本地模式（直接调用服务层）", file=sys.stderr)
        mcp.run(transport="stdio")
