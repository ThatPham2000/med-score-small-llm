from typing import List, Dict

import ollama

from llm import LLM


class LLMOllama(LLM):
    def __init__(self, model_name: str, ollama_client: ollama.AsyncClient = None):
        super().__init__(model_name)
        self.ollama_client = ollama_client

    async def generate(self, messages: List[Dict[str, str]]):
        if self.ollama_client is None:
            return ollama.AsyncClient.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "top_p": 0.1
                }
            )
        completion = self.ollama_client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": 0.3,
                "top_p": 0.1
            }
        )
        return completion

    def normalize_llm_response(self, completions) -> List[str]:
        return [c.message.content for c in completions]
