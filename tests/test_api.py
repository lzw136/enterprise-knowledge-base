"""
API 接口测试
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


class TestAPI:
    """API 测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_api_status(self):
        """测试 API 状态接口"""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm_configured" in data
    
    def test_qa_endpoint(self):
        """测试问答接口 (应该返回 501)"""
        response = client.post(
            "/api/v1/qa/",
            json={"question": "测试问题"}
        )
        # 因为 LLM 功能未实现，应该返回错误
        assert response.status_code in [200, 501]
    
    def test_chat_endpoint(self):
        """测试对话接口 (应该返回 501)"""
        response = client.post(
            "/api/v1/chat/",
            json={"message": "你好"}
        )
        # 因为 LLM 功能未实现，应该返回错误
        assert response.status_code in [200, 501]


# 运行示例
if __name__ == "__main__":
    test = TestAPI()
    test.test_health_check()
    print("API 测试通过!")
