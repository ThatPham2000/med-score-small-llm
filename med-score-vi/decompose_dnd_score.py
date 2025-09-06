import ast
from typing import List, Dict, Any

from decomposer import Decomposer
from prompts import DND_PROMPT
from llm import LLM


class DecomposeDnDScore(Decomposer):
    def __init__(
            self,
            llm: LLM = None,
    ):
        super().__init__(llm=llm)

        # Override self.agent to match settings from DnDScore
        # self.agent = partial(
        #     self.client.chat.completions.create,
        #     model=self.model_name,
        #     seed=self.random_state,
        #     temperature=0.75,
        #     top_p=1.0,
        #     max_tokens=2048
        # )

    def format_input(self, context: str, sentence: str) -> str:
        return DND_PROMPT.replace("[paragraph]", context).replace("[sentence]", sentence)

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        decompositions = []
        for d_input, completion in zip(decomp_input, completions):
            model_output = completion.strip()
            extra, subclaim_str = [x.strip() for x in model_output.split("##CONTEXT-SUBCLAIM PAIRS##:")]
            subclaim_str = subclaim_str.replace('\n', '').strip()
            subclaim_dict = ast.literal_eval(subclaim_str)
            explanation = extra.split("##EXPLANATION##:")[-1]

            decomp = {k: v for k, v in d_input.items() if k != "context"}

            # Error: malformed response
            if subclaim_dict is None:
                # logger.warning(f"Invalid dictionary. Skipping {d_input['id']=} {d_input['sentence_id']=}: {subclaim_dict=}")
                decomp["claim"] = None

            for idx, claim_dict in enumerate(subclaim_dict):
                decomp["claim"] = claim_dict["decontextualized"]
                decomp["claim_id"] = idx
                decomp["claim_meta"] = {
                    "subclaim": claim_dict["subclaim"],
                    "explanation": explanation
                }

            decompositions.append(decomp)
        return decompositions
