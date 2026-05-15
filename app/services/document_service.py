"""
文档服务实现
支持文件上传、解析、分块、向量入库、元数据持久化
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from app.services.rag_service import get_rag_service
from app.utils.text_splitter import chunk_documents


class DocumentService:
    """
    文档服务
    
    负责文档的上传、解析、分块、存储
    """

    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.upload_dir / "documents_metadata.json"
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()
        self.rag_service = get_rag_service()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                return {}
        return {}

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as fp:
            json.dump(self._metadata, fp, ensure_ascii=False, indent=2)

    async def upload_document(self, file, filename: str) -> Dict[str, Any]:
        content = await file.read()
        file_path = self.upload_dir / filename
        if file_path.exists():
            file_stem = file_path.stem
            file_suffix = file_path.suffix
            file_path = self.upload_dir / f"{file_stem}_{uuid.uuid4().hex}{file_suffix}"

        with open(file_path, "wb") as fp:
            fp.write(content)

        text = self.parse_document(str(file_path))
        chunks = self.split_documents(text)
        document_id = str(uuid.uuid4())

        metadata = {
            "id": document_id,
            "filename": file_path.name,
            "file_size": len(content),
            "file_type": file_path.suffix.lower().strip("."),
            "uploaded_at": datetime.now().isoformat(),
            "chunk_count": len(chunks),
        }

        self.rag_service.add_documents(chunks, {**metadata, "document_id": document_id})
        self._metadata[document_id] = metadata
        self._save_metadata()
        return metadata

    def parse_document(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                return fp.read()

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages)
            except Exception:
                return ""

        if suffix == ".docx":
            try:
                from docx import Document as DocxDocument
                document = DocxDocument(str(path))
                paragraphs = [p.text for p in document.paragraphs if p.text]
                return "\n".join(paragraphs)
            except Exception:
                return ""

        raise ValueError(f"不支持的文件类型: {suffix}")

    def split_documents(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text:
            return []
        return chunk_documents([text], chunk_size=chunk_size, overlap=overlap)

    def store_document(self, doc_id: str, chunks: List[str], metadata: Dict[str, Any]):
        self.rag_service.add_documents(chunks, {**metadata, "document_id": doc_id})

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self._metadata.values())

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._metadata:
            return False

        deleted = self.rag_service.delete_document(document_id)
        self._metadata.pop(document_id, None)
        self._save_metadata()
        return deleted
