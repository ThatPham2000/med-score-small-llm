import asyncio
from functools import partial
from typing import List, Dict

import backoff
import nest_asyncio
import requests
from openai import AsyncOpenAI

from llm import LLM

nest_asyncio.apply()


class LLMOpenAI(LLM):
    def __init__(self, model_name: str, server_path: str):
        super().__init__(model_name)
        self.client = AsyncOpenAI(base_url=server_path)
        self.agent = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            seed=42,
            temperature=0.0,
            top_p=1.0,
            max_tokens=256
        )

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=60
    )
    async def batch_response(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        async_responses = [
            self.agent(messages=x)
            for x in batch
        ]
        return await asyncio.gather(*async_responses)

    def normalize_llm_response(self, completions) -> List[str]:
        return [c.choices[0].message.content for c in completions]
