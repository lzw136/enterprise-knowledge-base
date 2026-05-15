"""
RAG 功能测试
"""

import pytest
from app.core.llm import LLMClient
from app.core.prompts import build_rag_prompt


class TestRAG:
    """RAG 测试类"""
    
    def test_rag_prompt_builder(self):
        """测试 RAG Prompt 构建"""
        question = "什么是 FastAPI?"
        context = [
            "FastAPI 是一个现代的 Python Web 框架",
            "它支持异步编程和自动文档生成"
        ]
        
        result = build_rag_prompt(question, context)
        
        assert "system" in result
        assert "user" in result
        assert question in result["user"]
        assert "FastAPI 是一个现代的 Python Web 框架" in result["user"]
    
    def test_llm_client_init(self):
        """测试 LLM 客户端初始化"""
        client = LLMClient(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model"
        )
        
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.test.com"
        assert client.model == "test-model"


# 运行示例
if __name__ == "__main__":
    test = TestRAG()
    test.test_rag_prompt_builder()
    print("RAG Prompt 测试通过!")
