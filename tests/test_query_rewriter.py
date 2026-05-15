"""
查询重写测试
"""

import pytest
from unittest.mock import Mock, patch
from app.core.query_rewriter import QueryRewriter, get_query_rewriter


class TestQueryRewriter:
    """查询重写器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.mock_llm = Mock()
        self.rewriter = QueryRewriter(llm_client=self.mock_llm)

    def test_rewrite_query(self):
        """测试查询重写"""
        self.mock_llm.chat.return_value = "优化后的查询"
        result = self.rewriter.rewrite_query("原始查询")
        assert result == "优化后的查询"

    def test_rewrite_query_empty(self):
        """测试空查询重写"""
        result = self.rewriter.rewrite_query("")
        assert result == ""

    def test_rewrite_query_same(self):
        """测试查询未改变"""
        self.mock_llm.chat.return_value = "原始查询"
        result = self.rewriter.rewrite_query("原始查询")
        assert result == "原始查询"

    def test_expand_query(self):
        """测试查询扩展"""
        self.mock_llm.chat.return_value = "扩展查询1\n扩展查询2\n扩展查询3"
        result = self.rewriter.expand_query("原始查询", num_expansions=3)
        assert len(result) == 4  # 原始查询 + 3个扩展
        assert result[0] == "原始查询"

    def test_expand_query_empty(self):
        """测试空查询扩展"""
        result = self.rewriter.expand_query("")
        assert result == [""]

    def test_decompose_query(self):
        """测试查询分解"""
        self.mock_llm.chat.return_value = "子问题1\n子问题2\n子问题3"
        result = self.rewriter.decompose_query("复杂问题")
        assert len(result) == 3

    def test_decompose_query_single(self):
        """测试单一查询分解"""
        self.mock_llm.chat.return_value = "原始问题"
        result = self.rewriter.decompose_query("简单问题")
        assert result == ["简单问题"]

    def test_rewrite_query_error(self):
        """测试查询重写错误处理"""
        self.mock_llm.chat.side_effect = Exception("API 错误")
        result = self.rewriter.rewrite_query("测试查询")
        assert result == "测试查询"

    def test_expand_query_error(self):
        """测试查询扩展错误处理"""
        self.mock_llm.chat.side_effect = Exception("API 错误")
        result = self.rewriter.expand_query("测试查询")
        assert result == ["测试查询"]

    def test_decompose_query_error(self):
        """测试查询分解错误处理"""
        self.mock_llm.chat.side_effect = Exception("API 错误")
        result = self.rewriter.decompose_query("测试查询")
        assert result == ["测试查询"]


class TestQueryRewriterIntegration:
    """查询重写器集成测试"""

    @patch('app.core.query_rewriter.get_llm_client')
    def test_get_query_rewriter(self, mock_get_llm):
        """测试获取查询重写器实例"""
        mock_get_llm.return_value = Mock()
        rewriter = get_query_rewriter()
        assert rewriter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
