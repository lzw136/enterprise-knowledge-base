"""
Agent 接口
基于 OpenAI 原生 Tool Calling 的智能助手
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.models.schemas import AgentResponse
from app.services.agent_service import get_agent_service
from app.services.rag_service import get_rag_service
from app.services.document_service import DocumentService

router = APIRouter()


class AgentRequest(BaseModel):
    """Agent 请求模型"""
    message: str
    max_steps: int = 5


@router.post("/", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Agent 智能助手接口

    基于 OpenAI 原生 Tool Calling，自动决策调用工具或直接回答：
    - search: 搜索知识库
    - qa: 基于知识库问答
    - list_documents: 列出已上传文档
    """
    try:
        rag_service = get_rag_service()
        document_service = DocumentService()
        agent = get_agent_service(
            rag_service=rag_service,
            document_service=document_service,
        )
        result = agent.run(
            user_message=request.message,
            max_steps=request.max_steps,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
