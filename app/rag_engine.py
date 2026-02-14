import os
from typing import List, Optional

class RAGEngine:
    def __init__(self):
        self._client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
        return self._client
    
    def generate_answer(self, query: str, context_chunks: List[dict]) -> dict:
        if not context_chunks:
            return {
                "answer": "Извините, я не нашёл релевантной информации в загруженных документах.",
                "confidence": 0.0
            }
        
        if not self.client:
            return self.generate_without_openai(query, context_chunks)
        
        context_text = "\n\n---\n\n".join([
            f"[Источник: {chunk['source']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        system_prompt = """Ты — интеллектуальный помощник для работы с документами. 
Отвечай на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте.
Отвечай на русском языке. Будь точным и информативным.
Если информации недостаточно, честно скажи об этом.
Всегда указывай источники в ответе."""

        user_prompt = f"""Контекст из документов:
{context_text}

Вопрос: {query}

Дай развёрнутый ответ на основе контекста. Укажи, какие источники использовал."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            avg_score = sum(c["score"] for c in context_chunks) / len(context_chunks)
            
            return {
                "answer": answer,
                "confidence": min(avg_score * 1.2, 1.0)
            }
        except Exception as e:
            return {
                "answer": f"Ошибка генерации ответа: {str(e)}",
                "confidence": 0.0
            }
    
    def generate_without_openai(self, query: str, context_chunks: List[dict]) -> dict:
        if not context_chunks:
            return {
                "answer": "Извините, я не нашёл релевантной информации в загруженных документах.",
                "confidence": 0.0
            }
        
        best_chunk = context_chunks[0]
        answer = f"Наиболее релевантный фрагмент из документа '{best_chunk['source']}':\n\n{best_chunk['content'][:500]}..."
        
        return {
            "answer": answer,
            "confidence": best_chunk["score"]
        }
