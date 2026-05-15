"""
对话接口
实现多轮对话功能
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from app.models.schemas import ChatMessage, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    """对话请求模型"""
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口
    
    支持多轮对话，可以传入 session_id 维持会话上下文
    """
    try:
        result = chat_service.chat(
            message=request.message,
            session_id=request.session_id,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in result.get("history", [])
        ]

        return {
            "session_id": result.get("session_id"),
            "response": result.get("response"),
            "message": {
                "role": "assistant",
                "content": result.get("response"),
            },
            "history": history,
        }
    except Exception as exc:
        raise HTTPException(status_code=501, detail=str(exc))
