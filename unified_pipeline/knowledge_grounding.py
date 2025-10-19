"""
Knowledge Grounding: Basic RAG (Retrieval-Augmented Generation)
Simple retrieval system for grounding LLM responses with external knowledge
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Evidence:
    """Represents a piece of retrieved evidence"""
    content: str
    source: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleRetriever:
    """
    Simple in-memory retriever for demonstration
    In production, replace with vector DB (ChromaDB, Pinecone, etc.)
    """
    
    def __init__(self, documents: Optional[List[Dict[str, str]]] = None):
        self.documents = documents or []
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the knowledge base"""
        self.documents.extend(documents)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Simple keyword-based retrieval
        In production, use semantic embeddings
        """
        
        # Simple scoring: count query term occurrences
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            
            # Count matching terms
            content_lower = content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            score = score / max(len(query_terms), 1)  # Normalize
            
            if score > 0:
                scored_docs.append((score, content, source))
        
        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        return [
            Evidence(content=content, source=source, relevance_score=score)
            for score, content, source in scored_docs[:top_k]
        ]
