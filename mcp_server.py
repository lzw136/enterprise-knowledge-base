"""
MCP Server - 企业智能知识库
将知识库暴露为 MCP 工具，供 Claude Desktop / Cursor 等客户端调用

使用方式:
  stdio 模式（默认）: python mcp_server.py
  SSE 模式:          python mcp_server.py sse
"""

import sys
import json
import uuid
from pathlib import Path
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP(
    "enterprise-knowledge-base",
    description="企业智能知识库 - 支持文档管理、语义搜索、智能问答",
)

# ==================== 服务层初始化（延迟加载） ====================

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
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在企业知识库中搜索相关信息。返回匹配的文档片段及相关度分数。

    Args:
        query: 搜索查询内容
        top_k: 返回结果数量，默认 5
    """
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

    Args:
        file_path: 本地文件的绝对路径
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"文件不存在: {file_path}"

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


# ==================== MCP 资源 ====================


@mcp.resource("knowledge://documents")
def get_documents_resource() -> str:
    """知识库文档列表资源"""
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

    if transport == "sse":
        from app.config import settings
        mcp.run(
            transport="sse",
            host=settings.mcp_host,
            port=settings.mcp_port,
        )
    else:
        mcp.run(transport="stdio")
