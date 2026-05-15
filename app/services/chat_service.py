"""
对话服务 (预留)
Day 4-5 实现多轮对话功能
"""

from typing import List, Dict, Optional
from app.core.llm import LLMClient, get_llm_client
from app.core.prompts import build_chat_messages


class ChatSession:
    """
    对话会话
    
    管理单个用户的对话历史
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            max_turns: 最多返回的对话轮数
            
        Returns:
            历史消息列表
        """
        return self.messages[-max_turns * 2:]
    
    def clear(self):
        """清空对话历史"""
        self.messages = []


class ChatService:
    """
    对话服务
    
    管理多个用户的对话会话
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化对话服务
        
        Args:
            llm_client: LLM 客户端实例
        """
        self.llm_client = llm_client or get_llm_client()
        self.sessions: Dict[str, ChatSession] = {}
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        """
        获取或创建会话
        
        Args:
            session_id: 会话 ID
            
        Returns:
            ChatSession 实例
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, any]:
        """
        对话接口
        
        Args:
            message: 用户消息
            session_id: 会话 ID (可选)
            system_prompt: 系统提示词 (可选)
            temperature: 温度参数
            max_tokens: 最大 token 数
            
        Returns:
            包含回复和会话信息的字典
        """
        # 获取或创建会话
        if session_id:
            session = self.get_or_create_session(session_id)
        else:
            # 使用临时会话
            session = ChatSession("temp")
        
        # 获取历史消息
        history = session.get_history()
        
        # 构建消息列表
        messages = build_chat_messages(
            user_message=message,
            system_prompt=system_prompt,
            history=history
        )
        
        # 调用 LLM
        response = self.llm_client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 更新会话历史
        session.add_user_message(message)
        session.add_assistant_message(response)
        
        return {
            "session_id": session.session_id,
            "response": response,
            "history": history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史
        
        Args:
            session_id: 会话 ID
            
        Returns:
            是否成功
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话 ID
            
        Returns:
            是否成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
