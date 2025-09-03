from typing import List, Dict


class LLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError