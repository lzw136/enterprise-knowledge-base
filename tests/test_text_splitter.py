"""
文本分块测试
"""

import pytest
from app.utils.text_splitter import TextSplitter, chunk_text, chunk_documents


class TestTextSplitter:
    """文本分块器测试类"""

    def test_split_fixed(self):
        """测试固定长度分块"""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2, strategy="fixed")
        text = "这是一段测试文本，用于测试固定长度分块功能。"
        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 12  # 允许稍微超过

    def test_split_sentence(self):
        """测试句子分块"""
        splitter = TextSplitter(chunk_size=50, strategy="sentence")
        text = "这是第一句话。这是第二句话！这是第三句话？"
        chunks = splitter.split_text(text)
        assert len(chunks) > 0

    def test_split_paragraph(self):
        """测试段落分块"""
        splitter = TextSplitter(chunk_size=100, strategy="paragraph")
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = splitter.split_text(text)
        assert len(chunks) == 3

    def test_split_recursive(self):
        """测试递归分块"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10, strategy="recursive")
        text = "这是一段较长的文本，需要进行递归分块处理。" * 10
        chunks = splitter.split_text(text)
        assert len(chunks) > 0

    def test_split_empty(self):
        """测试空文本分块"""
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_whitespace(self):
        """测试空白文本分块"""
        splitter = TextSplitter()
        chunks = splitter.split_text("   \n\t  ")
        assert chunks == []

    def test_split_short_text(self):
        """测试短文本分块"""
        splitter = TextSplitter(chunk_size=100)
        text = "短文本"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"

    def test_split_texts(self):
        """测试批量文本分块"""
        splitter = TextSplitter(chunk_size=50)
        texts = ["文本1", "文本2", "文本3"]
        chunks = splitter.split_texts(texts)
        assert len(chunks) == 3

    def test_chunk_text_function(self):
        """测试 chunk_text 函数"""
        text = "测试文本" * 20
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0

    def test_chunk_documents_function(self):
        """测试 chunk_documents 函数"""
        documents = ["文档1内容" * 10, "文档2内容" * 10]
        chunks = chunk_documents(documents, chunk_size=50, overlap=10)
        assert len(chunks) > 0

    def test_overlap(self):
        """测试重叠功能"""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5, strategy="fixed")
        text = "abcdefghijklmnopqrstuvwxyz" * 5
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_different_strategies(self):
        """测试不同策略"""
        text = "这是一段测试文本。" * 10

        splitter_fixed = TextSplitter(chunk_size=50, strategy="fixed")
        splitter_sentence = TextSplitter(chunk_size=50, strategy="sentence")
        splitter_paragraph = TextSplitter(chunk_size=50, strategy="paragraph")
        splitter_recursive = TextSplitter(chunk_size=50, strategy="recursive")

        chunks_fixed = splitter_fixed.split_text(text)
        chunks_sentence = splitter_sentence.split_text(text)
        chunks_paragraph = splitter_paragraph.split_text(text)
        chunks_recursive = splitter_recursive.split_text(text)

        # 所有策略都应该产生结果
        assert len(chunks_fixed) > 0
        assert len(chunks_sentence) > 0
        assert len(chunks_paragraph) > 0
        assert len(chunks_recursive) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
