import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="svod_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def add_chunks(self, chunks: List[dict]) -> int:
        if not chunks:
            return 0
        
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {"source": chunk["source"], "start": chunk["start"], "end": chunk["end"]}
            for chunk in chunks
        ]
        
        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": 1 - results["distances"][0][i]
            })
        
        return chunks
    
    def clear(self):
        self.client.delete_collection("svod_documents")
        self.collection = self.client.get_or_create_collection(
            name="svod_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        return self.collection.count()
