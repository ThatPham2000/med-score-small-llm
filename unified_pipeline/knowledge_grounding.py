import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from typing import Optional


@dataclass
class Evidence:
    content: str
    source: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleRetriever:

    def __init__(self, documents_path: str):
        # Load documents from a simple JSONL file
        if documents_path:
            import json
            with open(documents_path, "r", encoding="utf-8") as f:
                documents = [json.loads(line) for line in f]
            self.documents = documents or []
        else:
            self.documents = []

    def retrieve(self, query: str) -> List[Evidence]:
        question = self.get_question_from_prompt(query).strip()
        for document in self.documents:
            q = document.get("question", "").strip()
            if q and q.lower() == question.lower():
                return [Evidence(content=document.get("context", ""),
                                 source=document.get("uuid", "unknown"),
                                 relevance_score=1.0)]
        return []

    def get_question_from_prompt(self, prompt: str) -> Optional[str]:
        pattern = r"Question:\s*(.*?)\s*Based on the provided context"
        match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()
