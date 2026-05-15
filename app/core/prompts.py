"""
Prompt 模板 (预留)
Day 4-5 实现 RAG 和 Agent Prompt 模板
"""

from typing import List, Dict, Optional


# ==================== RAG 问答 Prompt ====================

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。

你的职责是根据提供的上下文信息，准确、清晰地回答用户的问题。

回答规则：
1. 只根据提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，诚实地告诉用户"我不知道"
3. 回答要简洁明了，使用中文
4. 适当引用上下文中的原文
5. 如果需要，可以对多个上下文片段进行综合整理
"""


RAG_USER_PROMPT_TEMPLATE = """请根据以下上下文信息回答用户的问题。

---
上下文：
{context}
---

用户问题：{question}

请给出回答："""


def build_rag_prompt(question: str, context: List[str]) -> Dict[str, str]:
    """
    构建 RAG 问答 Prompt
    
    Args:
        question: 用户问题
        context: 检索到的上下文片段列表
        
    Returns:
        包含 system 和 user 消息的字典
    """
    context_text = "\n---\n".join(context)
    
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )
    
    return {
        "system": RAG_SYSTEM_PROMPT,
        "user": user_prompt
    }


# ==================== 对话 Prompt ====================

DEFAULT_SYSTEM_PROMPT = """你是一个有帮助的 AI 助手。
请用友好、专业的方式回答用户的问题。
使用中文进行交流。"""


def build_chat_messages(
    user_message: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    构建对话消息列表
    
    Args:
        user_message: 当前用户消息
        system_prompt: 系统提示词
        history: 对话历史
        
    Returns:
        消息列表
    """
    messages = []
    
    # 添加系统提示词
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    
    # 添加历史消息
    if history:
        messages.extend(history)
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": user_message})
    
    return messages


# ==================== Agent Prompt ====================

AGENT_SYSTEM_PROMPT = """你是一个企业知识库智能助手。

你的职责是帮助用户查询企业知识库中的信息。你拥有工具来搜索知识库、回答问题和列出文档。

规则：
1. 当用户询问知识库中的具体信息时，优先使用 search 工具搜索相关内容
2. 当用户提出需要综合回答的问题时，使用 qa 工具
3. 当用户想了解知识库中有哪些文档时，使用 list_documents 工具
4. 如果用户的问题不需要工具（如闲聊、打招呼），直接回答
5. 始终使用中文回答
6. 基于工具返回的事实回答，不要编造信息"""
