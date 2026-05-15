"""
企业智能知识库问答系统 - FastAPI 应用入口

这是一个最简可运行的版本，Day 4-5 将完善 LLM 调用功能
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

# 创建 FastAPI 应用
app = FastAPI(
    title="企业智能知识库问答系统",
    description="基于 RAG + LLM 的企业级知识库问答系统",
    version="0.1.0",
    debug=settings.debug,
)

# ==================== CORS 配置 ====================
# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 健康检查接口 ====================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "enterprise-knowledge-base",
        "version": "0.1.0"
    }


# ==================== 根路径 ====================
@app.get("/")
async def root():
    """根路径 - 返回欢迎信息"""
    return {
        "message": "企业智能知识库问答系统 API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# ==================== 挂载路由 ====================
# 延迟导入，避免循环依赖
from app.api import documents, qa, chat, agent


@app.get("/api/v1/status")
async def api_status():
    """API 状态检查"""
    return {
        "status": "running",
        "llm_configured": bool(settings.openai_api_key),
        "model": settings.openai_model
    }


# 挂载路由 (预留接口)
app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
app.include_router(qa.router, prefix="/api/v1/qa", tags=["问答接口"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["对话接口"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent 智能助手"])


# ==================== 启动信息 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
