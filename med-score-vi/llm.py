from typing import List, Dict


class LLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, messages: List[Dict[str, str]]):
        raise NotImplementedError

    def normalize_llm_response(self, completions) -> List[str]:
        raise NotImplementedError
