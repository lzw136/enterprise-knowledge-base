"""
填充测试数据脚本
Day 6-7 使用
"""

from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# 示例文档数据
SAMPLE_DOCUMENTS = [
    {
        "title": "FastAPI 简介",
        "content": """
FastAPI 是一个现代、快速（高性能）的 Python Web 框架，基于标准 Python 类型提示。

主要特点：
1. 快速：极高的性能，可与 Node.js 和 Go 相媲美
2. 易用：自动生成 API 文档
3. 类型安全：基于 Pydantic 的数据验证
4. 异步支持：原生支持 async/await

FastAPI 常用于构建 AI 应用的后端 API，包括 LLM 调用、RAG 系统等。
        """.strip()
    },
    {
        "title": "RAG 技术介绍",
        "content": """
RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术。

工作流程：
1. 检索 (Retrieval): 从知识库中检索相关文档
2. 增强 (Augment): 将检索结果加入 Prompt
3. 生成 (Generate): LLM 基于增强后的 Prompt 生成答案

RAG 的优势：
- 可以利用最新的知识
- 可以访问私有数据
- 减少 LLM 幻觉
- 提高答案的准确性

常见的 RAG 组件包括：向量数据库、嵌入模型、检索算法等。
        """.strip()
    },
    {
        "title": "向量数据库选择",
        "content": """
常见的向量数据库包括：

1. ChromaDB
   - 轻量级，易于使用
   - 适合学习和原型开发
   - Python 原生支持

2. Milvus
   - 企业级，高性能
   - 支持分布式部署
   - 大规模向量检索

3. Pinecone
   - 云服务，无需运维
   - 托管式向量数据库
   - 适合生产环境

4. Weaviate
   - 开源，特性丰富
   - 支持混合搜索
   - GraphQL 接口

选择建议：
- 学习/原型：ChromaDB
- 中小规模：Pinecone
- 大规模/企业：Milvus
        """.strip()
    }
]


def seed_documents():
    """
    填充示例文档数据
    
    实际使用时需要结合 ChromaDB 进行向量存储
    """
    print("开始填充测试数据...")
    
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        print(f"\n文档 {i+1}: {doc['title']}")
        print(f"内容长度: {len(doc['content'])} 字符")
        # TODO: 实际存储到向量数据库
    
    print(f"\n共 {len(SAMPLE_DOCUMENTS)} 条测试数据")


if __name__ == "__main__":
    seed_documents()
    print("\n测试数据填充完成!")
