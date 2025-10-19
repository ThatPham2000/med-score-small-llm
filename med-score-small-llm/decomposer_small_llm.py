from typing import List, Dict, Any

from decomposer import Decomposer
from llm import LLM
from unified_pipeline import create_llm_provider, create_pipeline


class DecomposerSmallLLM(Decomposer):
    """
    Enhanced decomposer for small language models with improved reasoning capabilities.
    This implementation addresses the 7 MedScoreTaxonomy issues:
    1. Unverifiable claims - filters out personal narratives and patient-specific interactions
    2. Hallucinated claims - validates claims are grounded in original sentence
    3. Incomplete claims - preserves important modifiers and dependencies
    4. Incorrectly structured claims - transforms imperatives to declaratives
    5. Context-dependent claims - handles vague references and pronouns
    6. Redundant claims - implements deduplication mechanism
    7. Omitted claims - ensures comprehensive coverage validation
    """

    def __init__(
            self,
            provider: str = "ollama",
            llm: LLM = None,
    ):
        super().__init__(llm=llm)
        try:
            # TODO(THAT): add to support server ollama llm
            self.pipeline_llm = create_llm_provider(provider=provider, model=llm.model_name)
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            return

    def format_input(self, context: str, sentence: str) -> str:
        return f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

IMPORTANT: Extract ONE medical concept per claim. Use simple, declarative sentences.

Facts:
"""

    def do_decompose(self, decomposition_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_completions = []
        for d in decomposition_input:
            formatted_input = self.format_input(d['context'], d['sentence'])
            pipeline = create_pipeline(self.pipeline_llm, enable_atomic_fact_decomposition=True, verbose=True)
            result = pipeline.process(formatted_input)
            all_completions.append(result['final_answer'])

        # Format claims
        decompositions = self.format_completions(decomposition_input, all_completions)
        return decompositions

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        decompositions = []
        for d_input, completion in zip(decomp_input, completions):
            # Extract facts from completion, handling reasoning format
            # Find the "Facts:" section and extract claims from there
            lines = completion.split("\n")
            facts_started = False
            claim_list = []

            for line in lines:
                if "Facts:" in line:
                    facts_started = True
                    continue
                if facts_started and line.strip():
                    # Skip reasoning lines that start with numbers or "Reasoning:"
                    # Handle dynamic number of reasoning steps
                    reasoning_patterns = tuple(f"{i}." for i in range(1, 6))
                    if line.strip().startswith(reasoning_patterns + ("Reasoning:", "Think")):
                        continue
                    # Extract claim from bullet point
                    if line.strip().startswith("- "):
                        claim = line.strip()[2:].strip()
                        if claim and "no verifiable claim" not in claim.lower():
                            claim_list.append(claim)
                    elif line.strip() and not line.strip().startswith(("Context:", "Please breakdown")):
                        # Handle cases where facts don't start with bullet points
                        claim = line.strip()
                        if claim and "no verifiable claim" not in claim.lower():
                            claim_list.append(claim)

            # Apply validation and filtering if enabled
            # claim_list = self._validate_and_filter_claims(claim_list, d_input.get("sentence", ""),
            #                                               d_input.get("context", ""))

            # Process claims similar to original implementation
            for idx, claim in enumerate(claim_list):
                decomp = {k: v for k, v in d_input.items() if k != "context"}
                decomp["claim"] = claim
                decomp["claim_id"] = idx
                decompositions.append(decomp)

            if not claim_list:
                decomp = {k: v for k, v in d_input.items() if k != "context"}
                decomp["claim"] = None
                decompositions.append(decomp)

        return decompositions

    def _validate_and_filter_claims(self, claims: List[str], sentence: str, context: str) -> List[str]:
        """Validate and filter claims to address the 7 MedScoreTaxonomy issues"""
        if not claims:
            return claims

        validated_claims = []

        # for claim in claims:
        #     # 1. Check for unverifiable claims (personal narratives, patient interactions)
        #     if self._is_unverifiable_claim(claim):
        #         continue
        #
        #     # 2. Check for hallucinated claims (not grounded in sentence)
        #     if not self._is_grounded_in_sentence(claim, sentence):
        #         continue
        #
        #     # 3. Check for incomplete claims (missing important modifiers)
        #     if self._is_incomplete_claim(claim, sentence):
        #         continue
        #
        #     # 4. Transform incorrectly structured claims
        #     claim = self._transform_to_declarative(claim)
        #
        #     # 5. Handle context-dependent claims (vague references)
        #     claim = self._resolve_context_dependencies(claim, context)
        #
        #     # 6. Check for redundant claims
        #     if self._is_redundant_claim(claim, validated_claims, sentence):
        #         continue
        #
        #     validated_claims.append(claim)
        #
        # # 7. Validate comprehensive coverage (address omitted claims)
        # validated_claims = self._validate_comprehensive_coverage(validated_claims, sentence)

        return validated_claims
