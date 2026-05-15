"""
问答接口
实现 RAG 问答功能
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.schemas import QAResponse
from app.services.rag_service import get_rag_service

router = APIRouter()
rag_service = get_rag_service()


class QARequest(BaseModel):
    """问答请求模型"""
    question: str
    top_k: int = 5
    include_sources: bool = True


@router.post("/", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    问答接口
    
    基于知识库的问答，使用 RAG (Retrieval-Augmented Generation) 模式：
    1. 检索相关文档片段
    2. 构建 Prompt
    3. 调用 LLM 生成答案
    """
    try:
        result = rag_service.ask(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=501, detail=str(exc))
