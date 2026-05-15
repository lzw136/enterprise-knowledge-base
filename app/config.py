"""
配置管理模块
使用 pydantic-settings 读取环境变量，提供类型安全的配置访问
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """应用配置类"""
    
    # ==================== LLM 配置 ====================
    openai_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    openai_model: str = "qwen-plus"
    
    # ==================== Embedding 配置 ====================
    embedding_model: str = "text-embedding-3-small"
    
    # ==================== 向量数据库配置 ====================
    chroma_persist_directory: str = "./data/chroma_db"
    
    # ==================== 应用配置 ====================
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    
    # CORS 配置
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8501"]
    
    # ==================== Agent 配置 ====================
    use_langgraph_agent: bool = True

    # ==================== MCP Server 配置 ====================
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001

    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    
    class Config:
        """Pydantic 配置"""
        env_file = ".env"  # 自动从 .env 文件读取环境变量
        env_file_encoding = "utf-8"
        case_sensitive = False  # 环境变量不区分大小写


# 全局配置实例 (单例模式)
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例的依赖函数 (用于 FastAPI 依赖注入)"""
    return settings
