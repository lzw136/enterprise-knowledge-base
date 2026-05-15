"""
BM25 搜索引擎
基于传统关键词的检索方法，作为向量检索的补充
"""

import math
from typing import List, Dict, Any, Tuple
from collections import Counter
import jieba


class BM25Search:
    """
    BM25 搜索引擎
    
    BM25 是一种经典的信息检索算法，用于计算文档与查询的相关性得分
    常与向量检索配合使用，实现混合检索
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        初始化 BM25
        
        Args:
            k1: 词频饱和参数 (通常 1.2-2.0)
            b: 文档长度归一化参数 (通常 0.75)
        """
        self.k1 = k1
        self.b = b
        
        self.corpus: List[str] = []
        self.corpus_size: int = 0
        self.avgdl: float = 0
        self.doc_len: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_tokens: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
        # 使用 jieba 分词
        return list(jieba.cut(text))
    
    def fit(self, documents: List[str]):
        """
        构建索引
        
        Args:
            documents: 文档列表
        """
        self.corpus = documents
        self.corpus_size = len(documents)
        
        # 计算文档长度
        self.doc_len = []
        self.doc_freqs = []
        self.doc_tokens = []
        
        # IDF 统计
        df = Counter()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            
            # 统计词频
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            
            # 统计文档频率
            for word in freq:
                df[word] += 1
        
        # 计算平均文档长度
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # 计算 IDF
        self.idf = {}
        for word, freq in df.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[word] = idf
    
    def get_scores(self, query: str) -> List[float]:
        """
        获取所有文档的 BM25 得分
        
        Args:
            query: 查询文本
            
        Returns:
            每个文档的得分列表
        """
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        for i in range(self.corpus_size):
            scores[i] = self._calc_score(self.doc_freqs[i], query_tokens, self.doc_len[i])
        
        return scores
    
    def _calc_score(
        self,
        doc_freqs: Dict[str, int],
        query_tokens: List[str],
        doc_len: int,
    ) -> float:
        """
        计算单个文档的 BM25 得分
        
        Args:
            doc_freqs: 文档词频
            query_tokens: 查询词列表
            doc_len: 文档长度
            
        Returns:
            BM25 得分
        """
        score = 0.0
        
        for token in query_tokens:
            if token not in doc_freqs:
                continue
            
            freq = doc_freqs[token]
            idf = self.idf.get(token, 0)
            
            # BM25 公式
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * numerator / denominator
        
        return score
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            
        Returns:
            结果列表 [(文档索引, 得分, 文档内容), ...]
        """
        scores = self.get_scores(query)
        
        # 按得分排序
        results = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [(idx, score, self.corpus[idx]) for idx, score in results]
    
    def search_with_threshold(
        self,
        query: str,
        threshold: float = 0.1,
    ) -> List[Tuple[int, float, str]]:
        """
        带阈值的搜索
        
        Args:
            query: 查询文本
            threshold: 最低得分阈值
            
        Returns:
            结果列表
        """
        results = self.search(query, top_k=100)
        return [(idx, score, doc) for idx, score, doc in results if score >= threshold]


# 全局 BM25 实例 (延迟初始化)
_bm25: BM25Search = None


def get_bm25() -> BM25Search:
    """获取全局 BM25 实例"""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Search()
    return _bm25
