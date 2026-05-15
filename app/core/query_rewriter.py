"""
查询重写模块
优化用户查询，提高检索质量
"""

from typing import List, Optional
from app.core.llm import LLMClient, get_llm_client
from app.core.logger import app_logger


class QueryRewriter:
    """
    查询重写器

    使用 LLM 优化用户查询，包括：
    - 查询扩展：添加相关同义词
    - 查询澄清：消除歧义
    - 查询分解：将复杂问题拆分为子问题
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or get_llm_client()

    def rewrite_query(self, query: str) -> str:
        """
        重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询
        """
        if not query or not query.strip():
            return query

        prompt = f"""请优化以下搜索查询，使其更适合信息检索。

原始查询：{query}

优化要求：
1. 保持查询的核心意图不变
2. 添加相关的同义词或近义词
3. 如果查询模糊，尝试澄清
4. 使用简洁的中文表达

请只返回优化后的查询，不要添加其他解释："""

        try:
            messages = [
                {"role": "system", "content": "你是一个搜索查询优化专家。"},
                {"role": "user", "content": prompt}
            ]
            rewritten = self.llm_client.chat(messages, temperature=0.3, max_tokens=200)
            rewritten = rewritten.strip().strip('"').strip("'")

            if rewritten and rewritten != query:
                app_logger.info(f"查询重写: '{query}' -> '{rewritten}'")
                return rewritten
            return query
        except Exception as e:
            app_logger.warning(f"查询重写失败，使用原始查询: {e}")
            return query

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        查询扩展：生成多个相关查询

        Args:
            query: 原始查询
            num_expansions: 扩展数量

        Returns:
            扩展查询列表（包含原始查询）
        """
        if not query or not query.strip():
            return [query]

        prompt = f"""请为以下查询生成 {num_expansions} 个相关的搜索查询。

原始查询：{query}

要求：
1. 保持查询的核心意图
2. 使用不同的表达方式
3. 包含相关关键词
4. 每行一个查询

请直接返回查询列表："""

        try:
            messages = [
                {"role": "system", "content": "你是一个搜索查询扩展专家。"},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_client.chat(messages, temperature=0.5, max_tokens=300)

            # 解析扩展查询
            expansions = [line.strip() for line in response.split('\n') if line.strip()]
            expansions = [q.strip('-').strip('*').strip() for q in expansions]
            expansions = [q for q in expansions if q and q != query]

            # 限制数量
            expansions = expansions[:num_expansions]

            # 返回原始查询 + 扩展查询
            result = [query] + expansions
            app_logger.info(f"查询扩展: '{query}' -> {result}")
            return result
        except Exception as e:
            app_logger.warning(f"查询扩展失败: {e}")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        查询分解：将复杂问题拆分为子问题

        Args:
            query: 复杂查询

        Returns:
            子问题列表
        """
        if not query or not query.strip():
            return [query]

        prompt = f"""请将以下复杂问题分解为更简单的子问题。

原始问题：{query}

要求：
1. 每个子问题应该独立可答
2. 子问题的答案组合起来应该能回答原始问题
3. 每行一个问题

请直接返回子问题列表："""

        try:
            messages = [
                {"role": "system", "content": "你是一个问题分解专家。"},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_client.chat(messages, temperature=0.3, max_tokens=300)

            # 解析子问题
            sub_queries = [line.strip() for line in response.split('\n') if line.strip()]
            sub_queries = [q.strip('-').strip('*').strip('0123456789.').strip() for q in sub_queries]
            sub_queries = [q for q in sub_queries if q]

            if sub_queries and len(sub_queries) > 1:
                app_logger.info(f"查询分解: '{query}' -> {sub_queries}")
                return sub_queries
            return [query]
        except Exception as e:
            app_logger.warning(f"查询分解失败: {e}")
            return [query]


# 全局实例
_query_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """获取查询重写器实例"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter
