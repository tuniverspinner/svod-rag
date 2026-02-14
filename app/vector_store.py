import os
import json
import re
from typing import List
from collections import Counter
import math

class VectorStore:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_dir: str = "/tmp/svod_db"):
        if self._initialized:
            return
        
        self.persist_dir = persist_dir
        self.documents_file = os.path.join(persist_dir, "documents.json")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.documents = []
        self.doc_vectors = []
        self.vocab = {}
        self.idf = {}
        
        self._load()
        self._initialized = True
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 2]
    
    def _load(self):
        if os.path.exists(self.documents_file):
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self._compute_vectors()
    
    def _save(self):
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def _compute_vectors(self):
        if not self.documents:
            return
        
        doc_tokens = [self._tokenize(d["content"]) for d in self.documents]
        
        all_tokens = set()
        for tokens in doc_tokens:
            all_tokens.update(tokens)
        self.vocab = {t: i for i, t in enumerate(all_tokens)}
        
        N = len(self.documents)
        df = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1
        
        self.idf = {t: math.log(N / (1 + df[t])) + 1 for t in self.vocab}
        
        self.doc_vectors = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            total = sum(tf.values())
            vec = {}
            for t, count in tf.items():
                if t in self.vocab:
                    vec[t] = (count / total) * self.idf.get(t, 1)
            self.doc_vectors.append(vec)
    
    async def add_chunks(self, chunks: List[dict]) -> int:
        if not chunks:
            return 0
        
        for chunk in chunks:
            self.documents.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "source": chunk["source"],
                "start": chunk["start"],
                "end": chunk["end"]
            })
        
        self._compute_vectors()
        self._save()
        return len(chunks)
    
    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.documents or not self.doc_vectors:
            return []
        
        query_tokens = self._tokenize(query)
        tf = Counter(query_tokens)
        total = sum(tf.values())
        query_vec = {}
        for t, count in tf.items():
            if t in self.vocab:
                query_vec[t] = (count / total) * self.idf.get(t, 1)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            common_tokens = set(query_vec.keys()) & set(doc_vec.keys())
            if not common_tokens:
                scores.append(0)
                continue
            
            dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in common_tokens)
            mag_q = math.sqrt(sum(v ** 2 for v in query_vec.values()))
            mag_d = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
            
            if mag_q > 0 and mag_d > 0:
                scores.append(dot / (mag_q * mag_d))
            else:
                scores.append(0)
        
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:min(top_k, len(self.documents))]:
            if score > 0:
                results.append({
                    "content": self.documents[idx]["content"],
                    "source": self.documents[idx]["source"],
                    "score": float(score)
                })
        
        return results
    
    def clear(self):
        self.documents = []
        self.doc_vectors = []
        self.vocab = {}
        self.idf = {}
        self._save()
    
    def count(self) -> int:
        return len(self.documents)
