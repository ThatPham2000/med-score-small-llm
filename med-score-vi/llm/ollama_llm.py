from typing import List, Dict

from llm import LLM
import ollama


class OllamaLLM(LLM):
    def __init__(self, model_name: str, ollama_client: ollama.Client = None):
        super().__init__(model_name)
        self.ollama_client = ollama_client

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.ollama_client is None:
            completion = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "top_p": 0.1
                }
            )
            return completion.message.content
        completion = self.ollama_client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": 0.3,
                "top_p": 0.1
            }
        )
        return completion.message.content
