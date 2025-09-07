import asyncio
import string
from typing import List, Dict, Any

import nest_asyncio
from tqdm import tqdm

from utils import chunker

nest_asyncio.apply()

from llm import LLM


class Verifier(object):
    def __init__(
            self,
            llm: LLM = None,
            random_state: int = 42,
            batch_size: int = 32,
    ):
        self.llm = llm
        self.random_state = random_state
        self.batch_size = batch_size

    def do_verify(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verifier_inputs = self.add_evidence_to_verification_input(decompositions)
        messages = self.prepare_messages(verifier_inputs)

        all_completions = []
        n_iter = len(messages) // self.batch_size
        for batch in tqdm(chunker(messages, self.batch_size), desc="Verifier process", total=n_iter):
            completions = asyncio.run(self.llm.batch_response(batch))
            all_completions.extend(completions)

        verification_output = []
        for verifier_input, completion in zip(verifier_inputs, self.llm.normalize_llm_response(all_completions)):
            raw_output = completion.strip()
            is_supported = self.parse_verification_output(raw_output)
            output = {k: v for k, v in verifier_input.items()}
            output["raw"] = raw_output
            output["score"] = is_supported
            verification_output.append(output)
        return verification_output

    def parse_verification_output(self, completion_message: str) -> float:
        generated_answer = completion_message.strip().lower()
        is_supported = 0.0

        if "true" in generated_answer or "false" in generated_answer:
            if "true" in generated_answer and "false" not in generated_answer:
                is_supported = 1.0
            elif "false" in generated_answer and "true" not in generated_answer:
                is_supported = 0.0
            else:
                # Below logic is wrong: it gets the first occurrence of "true" and "false", and if the first true is later than the first false, it is considered true.
                # is_supported = generated_answer.index("true") > generated_answer.index("false")

                # If the last occurrence of 'true' appears later than 'false' in the output, then think the conclusion is true.
                is_supported = generated_answer.rindex("true") > generated_answer.rindex("false")
                is_supported = 1.0 if is_supported else 0.0
        else:
            generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
            is_supported = all(
                [keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
            is_supported = 1.0 if is_supported else 0.0
        return is_supported

    def add_evidence_to_verification_input(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_messages(self, verification_input: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError
