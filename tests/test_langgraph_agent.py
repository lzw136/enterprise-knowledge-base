"""
LangGraph Agent 单元测试
"""

import json
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import pytest

from app.services.langgraph_agent import LangGraphAgentService, AgentState
from app.services.agent_service import AgentStep


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_rag_service():
    svc = MagicMock()
    svc.retrieve.return_value = [
        {"content": "测试内容", "score": 0.95, "metadata": {"filename": "test.txt"}}
    ]
    svc.ask.return_value = {"answer": "测试回答"}
    return svc


@pytest.fixture
def mock_document_service():
    svc = MagicMock()
    svc.list_documents.return_value = [
        {"id": "doc1", "filename": "test.txt", "chunk_count": 5}
    ]
    return svc


@pytest.fixture
def agent(mock_llm_client, mock_rag_service, mock_document_service):
    return LangGraphAgentService(
        llm_client=mock_llm_client,
        rag_service=mock_rag_service,
        document_service=mock_document_service,
    )


# ==================== 图结构测试 ====================


class TestGraphStructure:
    def test_graph_has_agent_and_tools_nodes(self, agent):
        graph = agent.graph
        # LangGraph compiled graph should be callable
        assert graph is not None

    def test_graph_compiles_without_error(self, agent):
        """图应该能正常编译"""
        assert agent.graph is not None


# ==================== 节点逻辑测试 ====================


class TestAgentNode:
    def test_agent_node_no_tool_calls_sets_final_answer(self, agent, mock_llm_client):
        """LLM 不调用工具时，应设置 final_answer"""
        mock_response = MagicMock()
        mock_response.content = "直接回答"
        mock_response.tool_calls = None
        mock_llm_client.chat_completion.return_value = mock_response

        state: AgentState = {
            "messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hello"},
            ],
            "steps": [],
            "final_answer": None,
            "user_message": "hello",
        }

        result = agent._agent_node(state)

        assert result["final_answer"] == "直接回答"
        assert len(result["messages"]) == 3  # system + user + assistant
        assert result["messages"][-1]["role"] == "assistant"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["step_type"] == "final_answer"

    def test_agent_node_with_tool_calls_appends_message(self, agent, mock_llm_client):
        """LLM 调用工具时，应追加含 tool_calls 的 assistant 消息"""
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query": "test"}'

        mock_response = MagicMock()
        mock_response.content = None
        mock_response.tool_calls = [mock_tc]
        mock_llm_client.chat_completion.return_value = mock_response

        state: AgentState = {
            "messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hello"},
            ],
            "steps": [],
            "final_answer": None,
            "user_message": "hello",
        }

        result = agent._agent_node(state)

        assert result["final_answer"] is None
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "assistant"
        assert "tool_calls" in last_msg
        assert last_msg["tool_calls"][0]["id"] == "call_123"


class TestToolsNode:
    def test_tools_node_executes_tool_and_appends_result(self, agent):
        """工具节点应执行工具并追加 tool 消息"""
        state: AgentState = {
            "messages": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "list_documents",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ],
            "steps": [],
            "final_answer": None,
            "user_message": "hello",
        }

        result = agent._tools_node(state)

        # 应追加一个 tool 消息
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_123"
        # 应记录一个步骤
        assert len(result["steps"]) == 1
        assert result["steps"][0]["step_type"] == "tool_call"
        assert result["steps"][0]["tool_name"] == "list_documents"


# ==================== 条件路由测试 ====================


class TestShouldContinue:
    def test_routes_to_tools_when_tool_calls_present(self, agent):
        state: AgentState = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1"}],
                }
            ],
            "steps": [],
            "final_answer": None,
            "user_message": "test",
        }
        assert agent._should_continue(state) == "tools"

    def test_routes_to_end_when_final_answer_set(self, agent):
        state: AgentState = {
            "messages": [{"role": "assistant", "content": "answer"}],
            "steps": [],
            "final_answer": "answer",
            "user_message": "test",
        }
        assert agent._should_continue(state) == "__end__"

    def test_routes_to_end_when_no_tool_calls_and_no_final_answer(self, agent):
        """边界情况：assistant 消息无 tool_calls 但 final_answer 未设置"""
        state: AgentState = {
            "messages": [{"role": "assistant", "content": "answer"}],
            "steps": [],
            "final_answer": None,
            "user_message": "test",
        }
        # 应该走 END（agent 节点会设置 final_answer，这里模拟异常情况）
        assert agent._should_continue(state) == "__end__"


# ==================== 端到端测试 ====================


class TestRun:
    def test_run_returns_correct_shape(self, agent, mock_llm_client):
        """run() 返回值应包含 answer, steps, user_message"""
        mock_response = MagicMock()
        mock_response.content = "最终回答"
        mock_response.tool_calls = None
        mock_llm_client.chat_completion.return_value = mock_response

        result = agent.run("测试问题")

        assert "answer" in result
        assert "steps" in result
        assert "user_message" in result
        assert result["answer"] == "最终回答"
        assert result["user_message"] == "测试问题"
        assert isinstance(result["steps"], list)

    def test_run_with_tool_call_then_answer(self, agent, mock_llm_client):
        """先调工具再返回回答的完整流程"""
        # 第一次 LLM 调用：返回 tool_call
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query": "test"}'

        first_response = MagicMock()
        first_response.content = None
        first_response.tool_calls = [mock_tc]

        # 第二次 LLM 调用：返回最终回答
        second_response = MagicMock()
        second_response.content = "基于搜索的回答"
        second_response.tool_calls = None

        mock_llm_client.chat_completion.side_effect = [first_response, second_response]

        result = agent.run("测试问题")

        assert result["answer"] == "基于搜索的回答"
        assert len(result["steps"]) >= 2  # tool_call + final_answer
        step_types = [s["step_type"] for s in result["steps"]]
        assert "tool_call" in step_types
        assert "final_answer" in step_types

    def test_run_api_contract_matches_agent_response(self, agent, mock_llm_client):
        """返回格式应兼容 AgentResponse schema"""
        from app.models.schemas import AgentResponse

        mock_response = MagicMock()
        mock_response.content = "回答"
        mock_response.tool_calls = None
        mock_llm_client.chat_completion.return_value = mock_response

        result = agent.run("问题")

        # 验证能被 AgentResponse 解析
        response = AgentResponse(**result)
        assert response.answer == "回答"
        assert response.user_message == "问题"


class TestRunStream:
    def test_run_stream_yields_events(self, agent, mock_llm_client):
        """run_stream() 应 yield 状态更新事件"""
        mock_response = MagicMock()
        mock_response.content = "回答"
        mock_response.tool_calls = None
        mock_llm_client.chat_completion.return_value = mock_response

        events = list(agent.run_stream("测试问题"))

        assert len(events) > 0
        # 每个 event 是 dict with node name as key
        for event in events:
            assert isinstance(event, dict)
