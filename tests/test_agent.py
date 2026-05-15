"""
Agent 服务测试
"""

import pytest
from unittest.mock import Mock, patch
from app.services.agent_service import AgentService, AgentAction, get_agent_service


class TestAgentService:
    """Agent 服务测试类"""

    def setup_method(self):
        """测试前准备"""
        self.mock_llm = Mock()
        self.mock_rag = Mock()
        self.agent = AgentService(
            llm_client=self.mock_llm,
            rag_service=self.mock_rag
        )

    def test_agent_init(self):
        """测试 Agent 初始化"""
        assert self.agent.llm_client is not None
        assert self.agent.rag_service is not None
        assert len(self.agent.tools) > 0

    def test_agent_action_enum(self):
        """测试 Agent 动作枚举"""
        assert AgentAction.SEARCH == "search"
        assert AgentAction.QA == "qa"
        assert AgentAction.CHAT == "chat"
        assert AgentAction.UNKNOWN == "unknown"

    def test_register_tool(self):
        """测试工具注册"""
        mock_func = Mock(return_value="test result")
        self.agent.register_tool(
            name="test_tool",
            func=mock_func,
            description="测试工具"
        )
        assert "test_tool" in self.agent.tools

    def test_plan_search(self):
        """测试规划 - 搜索"""
        self.mock_llm.chat.return_value = "search"
        action = self.agent.plan("查询知识库中的信息")
        assert action == AgentAction.SEARCH

    def test_plan_qa(self):
        """测试规划 - 问答"""
        self.mock_llm.chat.return_value = "qa"
        action = self.agent.plan("什么是 FastAPI?")
        assert action == AgentAction.QA

    def test_plan_chat(self):
        """测试规划 - 对话"""
        self.mock_llm.chat.return_value = "chat"
        action = self.agent.plan("你好")
        assert action == AgentAction.CHAT

    def test_execute_search(self):
        """测试执行搜索"""
        self.mock_rag.retrieve.return_value = [
            {"content": "测试内容", "score": 0.9}
        ]
        result = self.agent.execute(AgentAction.SEARCH, {"user_message": "测试"})
        assert "测试内容" in result

    def test_execute_qa(self):
        """测试执行问答"""
        self.mock_rag.ask.return_value = {"answer": "测试答案"}
        result = self.agent.execute(AgentAction.QA, {"user_message": "测试问题"})
        assert result == "测试答案"

    def test_execute_chat(self):
        """测试执行对话"""
        result = self.agent.execute(AgentAction.CHAT, {"user_message": "你好"})
        assert result is None

    def test_run_complete(self):
        """测试完整运行流程"""
        # Mock LLM 响应
        self.mock_llm.chat.side_effect = [
            "思考：用户想查询信息\n行动：search\n行动输入：测试查询",
            "思考：找到信息\n行动：chat\n行动输入：这是答案"
        ]

        # Mock RAG 响应
        self.mock_rag.retrieve.return_value = [
            {"content": "相关信息", "score": 0.9}
        ]

        result = self.agent.run("测试查询")

        assert "answer" in result
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_parse_agent_response(self):
        """测试解析 Agent 响应"""
        response = "思考：分析用户需求\n行动：search\n行动输入：查询内容"
        thought, action, action_input = self.agent._parse_agent_response(response)

        assert thought == "分析用户需求"
        assert action == AgentAction.SEARCH
        assert action_input == "查询内容"

    def test_search_knowledge_base(self):
        """测试知识库搜索"""
        self.mock_rag.retrieve.return_value = [
            {"content": "测试内容", "score": 0.9, "metadata": {}}
        ]
        result = self.agent._search_knowledge_base("测试")
        assert "测试内容" in result

    def test_qa_knowledge_base(self):
        """测试知识库问答"""
        self.mock_rag.ask.return_value = {"answer": "测试答案"}
        result = self.agent._qa_knowledge_base("测试问题")
        assert result == "测试答案"

    def test_search_no_rag_service(self):
        """测试无 RAG 服务时的搜索"""
        agent = AgentService(llm_client=self.mock_llm, rag_service=None)
        result = agent._search_knowledge_base("测试")
        assert "未初始化" in result


class TestAgentIntegration:
    """Agent 集成测试"""

    @patch('app.services.agent_service.get_llm_client')
    @patch('app.services.agent_service.get_agent_service')
    def test_get_agent_service(self, mock_get_agent, mock_get_llm):
        """测试获取 Agent 服务实例"""
        mock_get_agent.return_value = AgentService()
        agent = get_agent_service()
        assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
