"""
嵌入模型封装 (预留)
Day 4-5 实现向量嵌入功能
"""

from typing import List, Optional, Union
import numpy as np


class Embeddings:
    """
    嵌入模型封装类
    
    支持 OpenAI 嵌入和本地 sentence-transformers
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化嵌入模型
        
        Args:
            model: 模型名称
            api_key: API Key
            base_url: API 基础地址
        """
        from app.config import settings
        
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_api_base
        self._client = None
        self._use_openai = bool(self.api_key)
    
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._use_openai:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            from sentence_transformers import SentenceTransformer
            self._client = SentenceTransformer(self.model)
        return self._client
    
    def embed_text(self, text: str) -> List[float]:
        """
        单个文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if not text:
            return []
        client = self._get_client()
        if self._use_openai:
            response = client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        else:
            embedding = client.encode(text, show_progress_bar=False)
            return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        client = self._get_client()
        if self._use_openai:
            response = client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        else:
            embeddings = client.encode(texts, show_progress_bar=False)
            return [vector.tolist() for vector in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            维度大小
        """
        # OpenAI text-embedding-3-small 默认 1536
        # sentence-transformers/all-MiniLM-L6-v2 是 384
        if "text-embedding-3-small" in self.model:
            return 1536
        elif "MiniLM" in self.model:
            return 384
        return 1536
