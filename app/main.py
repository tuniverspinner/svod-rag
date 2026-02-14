from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import tempfile
import uuid
from dotenv import load_dotenv

from app.models import QueryRequest, QueryResponse, UploadResponse, SourceChunk
from app.document_processor import DocumentLoader, TextChunker
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

load_dotenv()

app = FastAPI(
    title="Svod RAG API",
    description="RAG-as-a-Service для российских SMB",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()
rag_engine = RAGEngine()
chunker = TextChunker(chunk_size=500, overlap=50)

@app.get("/")
async def root():
    return {
        "service": "Svod RAG",
        "status": "operational",
        "documents": vector_store.count(),
        "version": "0.1.0"
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.txt', '.md']:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        text = DocumentLoader.extract_text(tmp_path)
        chunks = chunker.chunk(text, source=file.filename)
        
        if not chunks:
            return UploadResponse(
                document_id=str(uuid.uuid4()),
                chunks=0,
                status="empty_document"
            )
        
        vector_store.add_chunks(chunks)
        
        return UploadResponse(
            document_id=str(uuid.uuid4()),
            chunks=len(chunks),
            status="success"
        )
    finally:
        os.unlink(tmp_path)

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if vector_store.count() == 0:
        return QueryResponse(
            answer="Нет загруженных документов. Сначала загрузите документы через /api/upload",
            sources=[],
            confidence=0.0
        )
    
    context_chunks = vector_store.search(request.query, top_k=request.top_k)
    
    if os.getenv("OPENAI_API_KEY"):
        result = rag_engine.generate_answer(request.query, context_chunks)
    else:
        result = rag_engine.generate_without_openai(request.query, context_chunks)
    
    sources = [
        {
            "content": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
            "source": chunk["source"],
            "score": round(chunk["score"], 3)
        }
        for chunk in context_chunks[:3]
    ]
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        confidence=round(result["confidence"], 3)
    )

@app.post("/api/clear")
async def clear_documents():
    vector_store.clear()
    return {"status": "cleared", "documents": 0}

@app.get("/api/stats")
async def get_stats():
    return {
        "total_chunks": vector_store.count(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
