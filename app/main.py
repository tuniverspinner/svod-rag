from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv

from app.models import QueryRequest, QueryResponse, UploadResponse, SourceChunk
from app.document_processor import DocumentLoader, TextChunker
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

load_dotenv()

API_KEY = os.getenv("SVOD_API_KEY", "")
API_KEY_ENABLED = bool(API_KEY)

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not API_KEY_ENABLED:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

app = FastAPI(
    title="Svod RAG API",
    description="RAG-as-a-Service для российских SMB",
    version="0.1.1"
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

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Svod RAG - Demo not found</h1>", status_code=404)

@app.get("/api/health")
async def health():
    return {
        "service": "Svod RAG",
        "status": "operational",
        "documents": vector_store.count(),
        "version": "0.1.1"
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), _: bool = Depends(verify_api_key)):
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
        
        await vector_store.add_chunks(chunks)
        
        return UploadResponse(
            document_id=str(uuid.uuid4()),
            chunks=len(chunks),
            status="success"
        )
    finally:
        os.unlink(tmp_path)

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, _: bool = Depends(verify_api_key)):
    if vector_store.count() == 0:
        return QueryResponse(
            answer="Нет загруженных документов. Сначала загрузите документы через /api/upload",
            sources=[],
            confidence=0.0
        )
    
    context_chunks = await vector_store.search(request.query, top_k=request.top_k)
    
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
async def clear_documents(_: bool = Depends(verify_api_key)):
    vector_store.clear()
    return {"status": "cleared", "documents": 0}

@app.get("/api/stats")
async def get_stats():
    return {
        "total_chunks": vector_store.count(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
