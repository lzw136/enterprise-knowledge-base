"""
文本分块工具
支持多种分块策略：固定长度、句子、段落、递归分块
"""

import re
from typing import List, Callable, Optional
from app.core.logger import app_logger


class TextSplitter:
    """
    文本分块器

    支持多种分块策略:
    - 固定长度分块
    - 句子分块
    - 段落分块
    - 递归分块
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n",
        strategy: str = "recursive",
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每个块的目标大小 (字符数)
            chunk_overlap: 相邻块之间的重叠字符数
            separator: 分隔符
            strategy: 分块策略 (fixed/sentence/paragraph/recursive)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.strategy = strategy

        # 分隔符优先级列表（用于递归分块）
        self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " "]

    def split_text(self, text: str) -> List[str]:
        """
        分割文本

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []

        if self.strategy == "fixed":
            return self._split_fixed(text)
        elif self.strategy == "sentence":
            return self._split_sentence(text)
        elif self.strategy == "paragraph":
            return self._split_paragraph(text)
        elif self.strategy == "recursive":
            return self._split_recursive(text)
        else:
            app_logger.warning(f"未知分块策略: {self.strategy}，使用递归分块")
            return self._split_recursive(text)

    def _split_fixed(self, text: str) -> List[str]:
        """
        固定长度分块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # 尝试在分隔符处断开
            if end < len(text):
                last_newline = text.rfind("\n", start, end)
                if last_newline > start:
                    end = last_newline + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            # 移动起始位置 (考虑重叠)
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def _split_sentence(self, text: str) -> List[str]:
        """
        句子分块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        # 中英文句子分隔符
        sentence_endings = r'[。！？.!?；;]'
        sentences = re.split(f'({sentence_endings})', text)

        # 合并句子和标点
        merged_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            if sentence.strip():
                merged_sentences.append(sentence.strip())

        # 如果最后一个元素没有标点
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            merged_sentences.append(sentences[-1].strip())

        # 合并小句子到目标大小
        chunks = []
        current_chunk = ""

        for sentence in merged_sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_paragraph(self, text: str) -> List[str]:
        """
        段落分块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        # 按段落分割（双换行）
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个段落太长，递归分割
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self._split_recursive(paragraph)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_recursive(self, text: str) -> List[str]:
        """
        递归分块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割实现

        Args:
            text: 输入文本
            separators: 分隔符列表

        Returns:
            文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 找到合适的分隔符
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        # 按分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            # 没有分隔符，强制分割
            return self._split_fixed(text)

        # 合并小块
        chunks = []
        current_chunk = ""

        for split in splits:
            if not split.strip():
                continue

            # 如果当前块加上新块不超过目标大小
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)

                # 如果单个块太大，递归分割
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # 处理最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        # 添加重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        添加重叠

        Args:
            chunks: 文本块列表

        Returns:
            添加重叠后的文本块列表
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # 从前一个块的末尾取重叠部分
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # 尝试在句子边界处截断
            for sep in ["。", ".", "！", "!", "？", "?", "\n"]:
                last_sep = overlap_text.rfind(sep)
                if last_sep > 0:
                    overlap_text = overlap_text[last_sep + 1:]
                    break

            overlapped_chunks.append(overlap_text + current_chunk)

        return overlapped_chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        分割多个文本

        Args:
            texts: 文本列表

        Returns:
            合并后的文本块列表
        """
        all_chunks = []
        for text in texts:
            chunks = self.split_text(text)
            all_chunks.extend(chunks)
        return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "recursive",
) -> List[str]:
    """
    快捷函数: 文本分块

    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap: 重叠大小
        strategy: 分块策略

    Returns:
        文本块列表
    """
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        strategy=strategy
    )
    return splitter.split_text(text)


def chunk_documents(
    documents: List[str],
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "recursive",
) -> List[str]:
    """
    快捷函数: 批量文档分块

    Args:
        documents: 文档列表
        chunk_size: 块大小
        overlap: 重叠大小
        strategy: 分块策略

    Returns:
        文本块列表
    """
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        strategy=strategy
    )
    return splitter.split_texts(documents)
