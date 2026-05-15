"""
初始化数据库脚本
Day 6-7 使用
"""

import chromadb
from pathlib import Path


def init_chroma_db(persist_directory: str = "./data/chroma_db"):
    """
    初始化 ChromaDB
    
    Args:
        persist_directory: 持久化目录
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 ChromaDB 客户端
    client = chromadb.PersistentClient(path=str(persist_path))
    
    # 创建默认集合 (如果有需要)
    # collection = client.get_or_create_collection("documents")
    
    print(f"ChromaDB 已初始化: {persist_directory}")
    return client


def create_collections(client):
    """
    创建必要的集合
    
    Args:
        client: ChromaDB 客户端
    """
    # 文档集合
    doc_collection = client.get_or_create_collection(
        name="documents",
        metadata={"description": "企业文档知识库"}
    )
    
    # 问答对集合
    qa_collection = client.get_or_create_collection(
        name="qa_pairs",
        metadata={"description": "常见问答对"}
    )
    
    print("集合创建完成:")
    print(f"  - documents: {doc_collection.count()} 条")
    print(f"  - qa_pairs: {qa_collection.count()} 条")
    
    return doc_collection, qa_collection


if __name__ == "__main__":
    print("开始初始化数据库...")
    
    client = init_chroma_db()
    create_collections(client)
    
    print("数据库初始化完成!")
