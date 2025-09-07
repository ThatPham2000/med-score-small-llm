from typing import List, Dict

import backoff
import requests


class LLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def batch_response(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        raise NotImplementedError

    def normalize_llm_response(self, completions) -> List[str]:
        raise NotImplementedError
