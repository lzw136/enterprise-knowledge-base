"""
文档管理接口
实现文档上传、分块、存储和删除功能
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.models.schemas import DocumentResponse, DocumentListResponse
from app.services.document_service import DocumentService

router = APIRouter()
document_service = DocumentService()


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    获取文档列表
    """
    documents = document_service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.post("/", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档

    支持格式: txt, pdf, docx, md
    """
    try:
        metadata = await document_service.upload_document(file, file.filename)
        return DocumentResponse(**metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats")
async def get_stats():
    """获取知识库统计信息"""
    from app.services.rag_service import get_rag_service

    documents = document_service.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in documents)
    rag = get_rag_service()
    collection_count = rag.collection.count()

    return {
        "document_count": len(documents),
        "total_chunks": total_chunks,
        "collection_count": collection_count,
        "documents": documents,
    }


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    获取文档详情
    """
    doc = document_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentResponse(**doc)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    删除文档
    """
    if document_service.delete_document(document_id):
        return {"success": True, "message": "删除成功"}
    raise HTTPException(status_code=404, detail="文档不存在")
