"""
LLM 调用封装
支持 Qwen / DeepSeek 等 OpenAI 兼容接口
"""

import openai
from typing import List, Optional, Dict, Any, Generator, Union
from app.config import settings
from app.core.logger import app_logger


class LLMClient:
    """
    LLM 客户端封装类
    
    使用 OpenAI 兼容接口，支持 Qwen、DeepSeek 等模型
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API Key，默认从配置读取
            base_url: API 基础地址，默认从配置读取
            model: 模型名称，默认从配置读取
        """
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_api_base
        self.model = model or settings.openai_model
        self.client = None
        app_logger.info(f"LLM 客户端初始化，模型: {self.model}")
    
    def _get_client(self):
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise ValueError(
                "OpenAI API Key 未配置，请设置 OPENAI_API_KEY 或在配置中提供 API Key。"
            )
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self.client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        发送对话请求

        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            stream: 是否流式输出
            **kwargs: 其他参数

        Returns:
            LLM 生成的回复文本
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )

            if stream:
                # 流式输出：收集所有 chunks
                full_response = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                app_logger.debug(f"流式调用完成，响应长度: {len(full_response)}")
                return full_response
            else:
                return response.choices[0].message.content
        except Exception as e:
            app_logger.error(f"LLM 调用失败: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式对话请求（生成器版本）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            app_logger.error(f"LLM 流式调用失败: {e}")
            yield f"[错误: {str(e)}]"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Any:
        """
        发送对话请求，返回完整的消息对象（支持 tool_calls）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            tools: OpenAI 工具定义列表
            tool_choice: 工具选择策略 ("auto", "none", "required")
            **kwargs: 其他参数

        Returns:
            ChatCompletionMessage 对象，含 .content 和 .tool_calls 属性
        """
        try:
            client = self._get_client()
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            params.update(kwargs)

            response = client.chat.completions.create(**params)
            message = response.choices[0].message
            app_logger.debug(
                f"chat_completion 完成, "
                f"tool_calls: {bool(message.tool_calls)}, "
                f"content: {bool(message.content)}"
            )
            return message
        except Exception as e:
            app_logger.error(f"chat_completion 调用失败: {e}")
            raise

    def chat_with_system(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        带 System Prompt 的对话
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大 token 数
            
        Returns:
            LLM 回复
        """
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_message})
        
        return self.chat(messages, temperature, max_tokens)
    
    def batch_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_concurrency: int = 5,
    ) -> List[str]:
        """
        批量对话（并发请求）

        Args:
            messages_list: 多个对话消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_concurrency: 最大并发数

        Returns:
            回复列表
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        async def _chat(messages: List[Dict[str, str]]) -> str:
            return self.chat(messages, temperature, max_tokens)

        async def _batch_process():
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _limited_chat(messages):
                async with semaphore:
                    return await _chat(messages)

            tasks = [_limited_chat(m) for m in messages_list]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            results = asyncio.run(_batch_process())
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    app_logger.error(f"批量对话第 {i} 个请求失败: {result}")
                    processed_results.append(f"[错误: {str(result)}]")
                else:
                    processed_results.append(result)
            return processed_results
        except Exception as e:
            app_logger.error(f"批量对话执行失败: {e}")
            return [f"[错误: {str(e)}]" for _ in messages_list]


# 全局 LLM 客户端实例 (延迟初始化)
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端实例 (依赖注入模式)"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
