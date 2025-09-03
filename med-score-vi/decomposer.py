from typing import Optional

from llm import LLM


class Decomposer(object):
    def __init__(
            self,
            llm: LLM = None,
    ):
        self.llm = llm

    def get_system_prompt(self) -> Optional[str]:
        raise NotImplementedError

    def format_input(self, context: str, sentence: str) -> str:
        raise NotImplementedError
