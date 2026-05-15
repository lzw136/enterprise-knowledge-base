"""
RAG 服务实现
支持 ChromaDB 向量检索 + BM25 关键词检索
"""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from app.config import settings
from app.core.embeddings import Embeddings
from app.core.llm import LLMClient, get_llm_client
from app.core.prompts import build_rag_prompt
from app.core.logger import app_logger
from app.core.query_rewriter import QueryRewriter, get_query_rewriter
from app.utils.bm25 import BM25Search
from app.utils.cache import cached, cache_manager
from app.models.schemas import SourceDocument


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 服务
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, top_k: int = 5):
        self.llm_client = llm_client or get_llm_client()
        self.top_k = top_k
        self.embeddings = Embeddings()
        self.bm25 = BM25Search()
        self.query_rewriter = get_query_rewriter()

        # 使用新版 ChromaDB API
        persist_dir = Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir)
        )
        app_logger.info(f"ChromaDB 初始化完成，存储路径: {persist_dir}")

        self.collection = self._get_or_create_collection("documents")
        self._chunk_texts: List[str] = []
        self._chunk_ids: List[str] = []
        self._chunk_metadata: List[Dict[str, Any]] = []
        self._refresh_collection_state()

    def _get_or_create_collection(self, name: str):
        try:
            collection = self.client.get_collection(name)
            app_logger.info(f"获取已有集合: {name}, 文档数: {collection.count()}")
            return collection
        except Exception as e:
            app_logger.warning(f"集合 {name} 不存在，创建新集合: {e}")
            return self.client.create_collection(name=name)

    def _refresh_collection_state(self):
        try:
            state = self.collection.get(include=["documents", "metadatas"])
            self._chunk_texts = state.get("documents", []) or []
            self._chunk_metadata = state.get("metadatas", []) or []
            self._chunk_ids = state.get("ids", []) or []
            self.bm25.fit(self._chunk_texts)
            app_logger.debug(f"刷新集合状态完成，文档块数: {len(self._chunk_texts)}")
        except Exception as e:
            app_logger.error(f"刷新集合状态失败: {e}")
            raise

    def _query_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not query or not self._chunk_texts:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self._chunk_texts)),
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            output = []
            for content, metadata, distance, chunk_id in zip(documents, metadatas, distances, ids):
                score = float(1.0 / (distance + 1e-6)) if distance is not None else 0.0
                output.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                })

            app_logger.debug(f"向量检索完成，查询: {query[:50]}..., 结果数: {len(output)}")
            return output
        except Exception as e:
            app_logger.error(f"向量检索失败: {e}")
            return []

    def _query_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self._chunk_texts:
            return []

        try:
            results = self.bm25.search(query, top_k=top_k)
            output = []
            for idx, score, content in results:
                metadata = self._chunk_metadata[idx] if idx < len(self._chunk_metadata) else {}
                chunk_id = self._chunk_ids[idx] if idx < len(self._chunk_ids) else None
                output.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "score": float(score),
                })

            app_logger.debug(f"BM25 检索完成，查询: {query[:50]}..., 结果数: {len(output)}")
            return output
        except Exception as e:
            app_logger.error(f"BM25 检索失败: {e}")
            return []

    def retrieve(self, query: str, top_k: Optional[int] = None, use_rewrite: bool = True) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k

        # 查询重写（可选）
        search_query = query
        if use_rewrite:
            try:
                search_query = self.query_rewriter.rewrite_query(query)
            except Exception as e:
                app_logger.warning(f"查询重写失败，使用原始查询: {e}")

        # 混合检索
        vector_results = self._query_vector(search_query, top_k)
        bm25_results = self._query_bm25(search_query, top_k)

        # 融合检索结果
        fused: Dict[str, Dict[str, Any]] = {}
        for item in vector_results + bm25_results:
            key = item["id"] or item["content"]
            if key not in fused:
                fused[key] = item.copy()
            else:
                # 同一文档块的分数累加
                fused[key]["score"] += item["score"]

        results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        app_logger.info(f"混合检索完成，查询: {query[:50]}..., 返回结果数: {len(results)}")
        return results

    def generate(self, question: str, context: List[str], temperature: float = 0.7) -> str:
        prompt = build_rag_prompt(question, context)
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return self.llm_client.chat(messages, temperature=temperature)

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        top_k = top_k or self.top_k
        sources: List[SourceDocument] = []

        app_logger.info(f"开始问答流程，问题: {question}")

        # 检索相关文档
        candidates = self.retrieve(question, top_k=top_k)
        context = [item["content"] for item in candidates]

        if not context:
            app_logger.warning(f"未找到相关文档，问题: {question}")
            return {
                "answer": "抱歉，未找到与您问题相关的信息。",
                "question": question,
                "sources": [],
                "model": self.llm_client.model,
                "tokens_used": None,
            }

        # 生成回答
        answer = self.generate(question, context, temperature=temperature)
        app_logger.info(f"问答完成，问题: {question[:50]}...")

        # 构建来源文档
        if include_sources:
            for item in candidates:
                metadata = item.get("metadata", {}) or {}
                sources.append(SourceDocument(
                    chunk_id=item.get("id", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                    document_id=metadata.get("document_id"),
                    document_name=metadata.get("filename"),
                ))

        return {
            "answer": answer,
            "question": question,
            "sources": sources if include_sources else None,
            "model": self.llm_client.model,
            "tokens_used": None,
        }

    def add_documents(self, chunks: List[str], metadata: Dict[str, Any]):
        if not chunks:
            app_logger.warning("尝试添加空文档块")
            return

        document_id = metadata.get("document_id") or metadata.get("id") or str(uuid.uuid4())
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{idx}"
            ids.append(chunk_id)
            metadatas.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                **metadata,
            })

        try:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            self._refresh_collection_state()
            app_logger.info(f"成功添加文档，文档ID: {document_id}, 块数: {len(chunks)}")
        except Exception as e:
            app_logger.error(f"添加文档失败: {e}")
            raise

    def delete_document(self, document_id: str) -> bool:
        try:
            matched_ids = [meta.get("chunk_id") for meta in self._chunk_metadata if meta.get("document_id") == document_id]
            if matched_ids:
                self.collection.delete(ids=matched_ids)
                app_logger.info(f"删除文档 {document_id} 的 {len(matched_ids)} 个块")
            else:
                self.collection.delete(where={"document_id": document_id})
                app_logger.info(f"通过条件删除文档: {document_id}")

            self._refresh_collection_state()
            return True
        except Exception as e:
            app_logger.error(f"删除文档失败: {e}")
            return False


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
