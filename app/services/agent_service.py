"""
Agent 服务
基于 OpenAI 原生 Tool Calling 实现智能助手
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
import json

from app.core.llm import LLMClient, get_llm_client
from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.core.logger import app_logger


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def openai_tool_schema(self) -> Dict[str, Any]:
        """生成 OpenAI Function Calling 的 JSON Schema"""
        properties = {}
        required = []
        for param_name, param_desc in (self.parameters or {}).items():
            properties[param_name] = {
                "type": "string",
                "description": str(param_desc),
            }
            required.append(param_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class AgentStep:
    """Agent 执行步骤"""
    step_type: str                          # "tool_call" or "final_answer"
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    tool_result: Optional[str] = None
    content: Optional[str] = None


class AgentService:
    """
    Agent 服务

    基于 OpenAI 原生 Tool Calling 实现 ReAct 循环：
    1. 将用户消息 + 工具定义发送给 LLM
    2. LLM 决定是调用工具还是直接回答
    3. 如果调用工具：执行工具，将结果回传 LLM，循环继续
    4. 如果直接回答：返回最终答案
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
        app_logger.info("Agent 服务初始化完成")

    def _register_default_tools(self):
        """注册默认工具"""
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
        app_logger.info(f"注册工具: {name}")

    # ==================== 工具实现 ====================

    def _search_knowledge_base(self, query: str) -> str:
        """搜索知识库"""
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
        """基于知识库问答"""
        if not self.rag_service:
            return "知识库服务未初始化"
        try:
            result = self.rag_service.ask(question, include_sources=False)
            return result.get("answer", "无法生成回答")
        except Exception as e:
            app_logger.error(f"知识库问答失败: {e}")
            return f"问答失败: {str(e)}"

    def _list_documents(self) -> str:
        """列出知识库文档"""
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

    # ==================== Agent 核心循环 ====================

    def run(self, user_message: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        运行 Agent

        Args:
            user_message: 用户消息
            max_steps: 最大执行步骤数

        Returns:
            {"answer": str, "steps": list, "user_message": str}
        """
        steps: List[AgentStep] = []
        final_answer = None

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = [tool.openai_tool_schema for tool in self.tools.values()]

        app_logger.info(f"Agent 开始处理: {user_message}")

        for iteration in range(max_steps):
            try:
                response_msg = self.llm_client.chat_completion(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=2000,
                )
            except Exception as e:
                app_logger.error(f"Agent LLM 调用失败: {e}")
                final_answer = f"抱歉，AI 服务调用失败: {str(e)}"
                break

            # 情况1：无工具调用 → 最终回答
            if not response_msg.tool_calls:
                final_answer = response_msg.content or ""
                steps.append(AgentStep(
                    step_type="final_answer",
                    content=final_answer,
                ))
                break

            # 情况2：有工具调用 → 执行工具并继续循环
            # 将 assistant 消息（含 tool_calls）追加到对话历史
            messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_msg.tool_calls
                ],
            })

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments
                tool_call_id = tool_call.id

                # 解析参数
                try:
                    func_args = json.loads(func_args_str)
                except json.JSONDecodeError:
                    func_args = {}

                # 执行工具
                if func_name in self.tools:
                    try:
                        tool_func = self.tools[func_name].func
                        result = tool_func(**func_args)
                    except Exception as e:
                        result = f"工具执行失败: {str(e)}"
                        app_logger.error(f"工具 {func_name} 执行失败: {e}")
                else:
                    result = f"未知工具: {func_name}"

                # 记录步骤
                steps.append(AgentStep(
                    step_type="tool_call",
                    tool_name=func_name,
                    tool_arguments=func_args,
                    tool_call_id=tool_call_id,
                    tool_result=str(result),
                ))

                # 将工具结果追加到对话历史（OpenAI 协议要求 tool_call_id 匹配）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                })

            app_logger.info(
                f"Agent 步骤 {iteration + 1}: "
                f"调用了 {[tc.function.name for tc in response_msg.tool_calls]}"
            )

        if final_answer is None:
            final_answer = "抱歉，Agent 未能在限定步骤内生成回答。"

        app_logger.info(f"Agent 处理完成，步骤数: {len(steps)}")

        return {
            "answer": final_answer,
            "steps": [asdict(s) for s in steps],
            "user_message": user_message,
        }


# 全局 Agent 服务实例
_agent_service: Optional[AgentService] = None


def get_agent_service(rag_service=None, document_service=None):
    """获取 Agent 服务实例（根据配置返回 LangGraph 或原版实现）"""
    global _agent_service
    if _agent_service is None:
        from app.config import settings
        if settings.use_langgraph_agent:
            from app.services.langgraph_agent import LangGraphAgentService
            _agent_service = LangGraphAgentService(
                rag_service=rag_service,
                document_service=document_service,
            )
        else:
            _agent_service = AgentService(
                rag_service=rag_service,
                document_service=document_service,
            )
    return _agent_service
