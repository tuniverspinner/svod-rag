import os
import tempfile
from typing import List, Optional
from PyPDF2 import PdfReader

class DocumentLoader:
    @staticmethod
    def extract_text(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return DocumentLoader._extract_pdf(file_path)
        elif ext in ['.txt', '.md']:
            return DocumentLoader._extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    @staticmethod
    def _extract_text(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str = "unknown") -> List[dict]:
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "id": f"{source}_{chunk_id}",
                    "content": chunk_text,
                    "source": source,
                    "start": start,
                    "end": min(end, len(text))
                })
                chunk_id += 1
            
            start = end - self.overlap
        
        return chunks
