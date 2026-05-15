"""
LangGraph Agent 服务
基于 LangGraph StateGraph 实现智能助手，替代手动 ReAct 循环

Graph topology:
    START → "agent" ──(有 tool_calls)──→ "tools" → "agent"
               │
               └──(无 tool_calls)──→ END
"""

from typing import TypedDict, List, Dict, Any, Optional, Callable
from dataclasses import asdict
import json

from langgraph.graph import StateGraph, END

from app.core.llm import LLMClient, get_llm_client
from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.core.logger import app_logger
from app.services.agent_service import ToolDefinition, AgentStep


class AgentState(TypedDict):
    """LangGraph Agent 共享状态"""
    messages: List[Dict[str, Any]]      # OpenAI 格式消息历史
    steps: List[Dict[str, Any]]         # AgentStep 字典列表
    final_answer: Optional[str]         # 最终回答
    user_message: str                   # 原始用户消息


class LangGraphAgentService:
    """
    基于 LangGraph 的 Agent 服务

    与 AgentService 保持相同的 run() 接口，可无缝替换。
    支持 run() 同步执行和 run_stream() 流式执行。
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_service=None,
        document_service=None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.rag_service = rag_service
        self.document_service = document_service
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
        self.graph = self._build_graph()
        app_logger.info("LangGraph Agent 服务初始化完成")

    # ==================== 工具注册 ====================

    def _register_default_tools(self):
        """注册默认工具（与 AgentService 一致）"""
        self.register_tool(
            name="search",
            func=self._search_knowledge_base,
            description="在企业知识库中搜索相关信息",
            parameters={"query": "搜索查询内容"},
        )
        self.register_tool(
            name="qa",
            func=self._qa_knowledge_base,
            description="基于知识库回答问题",
            parameters={"question": "要回答的问题"},
        )
        self.register_tool(
            name="list_documents",
            func=self._list_documents,
            description="列出知识库中所有已上传的文档",
            parameters={},
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any] = None,
    ):
        """注册工具"""
        self.tools[name] = ToolDefinition(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {},
        )
        app_logger.info(f"[LangGraph] 注册工具: {name}")

    # ==================== 工具实现 ====================

    def _search_knowledge_base(self, query: str) -> str:
        if not self.rag_service:
            return "知识库服务未初始化"
        try:
            results = self.rag_service.retrieve(query, top_k=3)
            if not results:
                return "未找到相关信息"
            formatted = []
            for i, result in enumerate(results, 1):
                content = result.get("content", "")[:200]
                score = result.get("score", 0)
                formatted.append(f"{i}. [{score:.2f}] {content}...")
            return "\n".join(formatted)
        except Exception as e:
            app_logger.error(f"知识库搜索失败: {e}")
            return f"搜索失败: {str(e)}"

    def _qa_knowledge_base(self, question: str) -> str:
        if not self.rag_service:
            return "知识库服务未初始化"
        try:
            result = self.rag_service.ask(question, include_sources=False)
            return result.get("answer", "无法生成回答")
        except Exception as e:
            app_logger.error(f"知识库问答失败: {e}")
            return f"问答失败: {str(e)}"

    def _list_documents(self) -> str:
        if not self.document_service:
            return "文档服务未初始化"
        try:
            docs = self.document_service.list_documents()
            if not docs:
                return "知识库中暂无文档"
            return json.dumps(docs, ensure_ascii=False, default=str)
        except Exception as e:
            app_logger.error(f"列出文档失败: {e}")
            return f"获取文档列表失败: {str(e)}"

    # ==================== 图构建 ====================

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)

        # 设置入口
        workflow.set_entry_point("agent")

        # 条件边：agent → tools 或 END
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        # tools → agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # ==================== 图节点 ====================

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 节点：调用 LLM，决定是调用工具还是返回最终回答"""
        messages = state["messages"]
        steps = list(state["steps"])

        tool_schemas = [t.openai_tool_schema for t in self.tools.values()]

        try:
            response_msg = self.llm_client.chat_completion(
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            app_logger.error(f"[LangGraph] Agent LLM 调用失败: {e}")
            return {
                "messages": messages,
                "steps": steps,
                "final_answer": f"抱歉，AI 服务调用失败: {str(e)}",
            }

        # 构建 assistant 消息
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response_msg.content,
        }

        if response_msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_msg.tool_calls
            ]

        new_messages = messages + [assistant_msg]

        # 无 tool_calls → 最终回答
        final_answer = None
        if not response_msg.tool_calls:
            final_answer = response_msg.content or ""
            steps.append(asdict(AgentStep(
                step_type="final_answer",
                content=final_answer,
            )))

        return {
            "messages": new_messages,
            "steps": steps,
            "final_answer": final_answer,
        }

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        """工具节点：执行所有 tool_calls"""
        messages = list(state["messages"])
        steps = list(state["steps"])

        last_msg = messages[-1]
        tool_calls = last_msg.get("tool_calls", [])

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            tool_call_id = tc["id"]

            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                func_args = {}

            if func_name in self.tools:
                try:
                    result = self.tools[func_name].func(**func_args)
                except Exception as e:
                    result = f"工具执行失败: {str(e)}"
                    app_logger.error(f"[LangGraph] 工具 {func_name} 执行失败: {e}")
            else:
                result = f"未知工具: {func_name}"

            steps.append(asdict(AgentStep(
                step_type="tool_call",
                tool_name=func_name,
                tool_arguments=func_args,
                tool_call_id=tool_call_id,
                tool_result=str(result),
            )))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result),
            })

        tool_names = [tc["function"]["name"] for tc in tool_calls]
        app_logger.info(f"[LangGraph] 工具节点执行完成: {tool_names}")

        return {"messages": messages, "steps": steps}

    def _should_continue(self, state: AgentState) -> str:
        """条件路由：有 tool_calls → tools，否则 → END"""
        if state.get("final_answer") is not None:
            return END
        last_msg = state["messages"][-1]
        if last_msg.get("tool_calls"):
            return "tools"
        return END

    # ==================== 执行接口 ====================

    def run(self, user_message: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        运行 Agent（同步）。与 AgentService.run() 接口一致。

        Args:
            user_message: 用户消息
            max_steps: 最大执行步骤数

        Returns:
            {"answer": str, "steps": list, "user_message": str}
        """
        initial_state: AgentState = {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "steps": [],
            "final_answer": None,
            "user_message": user_message,
        }

        # recursion_limit: 每轮 agent→tools 算 2 步，最后 agent→END 算 1 步
        config = {"recursion_limit": max_steps * 2 + 1}

        app_logger.info(f"[LangGraph] Agent 开始处理: {user_message}")

        final_state = self.graph.invoke(initial_state, config=config)

        answer = final_state.get("final_answer") or "抱歉，Agent 未能在限定步骤内生成回答。"
        steps = final_state.get("steps", [])

        app_logger.info(f"[LangGraph] Agent 处理完成，步骤数: {len(steps)}")

        return {
            "answer": answer,
            "steps": steps,
            "user_message": user_message,
        }

    def run_stream(self, user_message: str, max_steps: int = 5):
        """
        流式运行 Agent（生成器）。每次节点执行后 yield 状态更新。

        适用于 UI 实时展示推理过程。

        Yields:
            Dict[str, AgentState] - 节点名到状态更新的映射
        """
        initial_state: AgentState = {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "steps": [],
            "final_answer": None,
            "user_message": user_message,
        }

        config = {"recursion_limit": max_steps * 2 + 1}

        app_logger.info(f"[LangGraph] Agent 流式处理开始: {user_message}")

        for event in self.graph.stream(initial_state, config=config):
            yield event
