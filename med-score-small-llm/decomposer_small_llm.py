from typing import List, Dict, Any

from decomposer import Decomposer
from llm import LLM


class DecomposerSmallLLM(Decomposer):
    """
    Enhanced decomposer for small language models with improved reasoning capabilities.
    This implementation makes small language models more intelligent by:
    1. Using chain-of-thought prompting
    2. Implementing multi-step reasoning
    3. Adding context-aware decomposition
    4. Using few-shot examples with reasoning
    """

    def __init__(
            self,
            llm: LLM = None,
            reasoning_steps: int = 3,
    ):
        super().__init__(llm=llm)
        self.reasoning_steps = reasoning_steps

    def get_system_prompt(self) -> str:
        """Enhanced system prompt with chain-of-thought reasoning for small LLMs"""
        # Generate dynamic reasoning steps based on the reasoning_steps parameter
        reasoning_steps_text = self._generate_reasoning_steps()
        reasoning_format = self._generate_reasoning_format()

        # Generate dynamic examples based on reasoning steps
        examples = self._generate_examples()

        return f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning.

REASONING PROCESS:
{reasoning_steps_text}

The facts should be objective and verifiable against reliable external information such as Wikipedia and PubMed. All subjective personal experiences ("I was or someone did") and personal narratives (stating a past event) are not verifiable and should not be included in the fact list. Facts should be situated within conditions in the sentence. Suggestions (e.g. "I recommend or Your doctor suggest") and opinions (e.g. "I think") should be transformed into objective facts by removing subjective words and pronouns to only retain the core information that can be verified. Imperative instructions ("do something") should be transformed into declarative facts ("doing something is helpful for some conditions").

If there is an overly specific entity such as "Your partner" or vague references (pronouns, this or that) in the fact, replace it with a general phrase with conditional modifiers using information in the provided context (e.g. "People in some conditions"). Each fact should be verifiable on its own and require no additional context. Do not add additional information outside of the sentence and context.

REASONING FORMAT:
Think step by step:
{reasoning_format}

If there is no verifiable fact in the sentence, please write "No verifiable claim".

Here are some examples with reasoning:

{examples}

Now, for your task, follow the same reasoning process."""

    def format_input(self, context: str, sentence: str) -> str:
        """Enhanced input formatting with reasoning prompts"""
        reasoning_format = self._generate_reasoning_format()
        return f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

Reasoning:
{reasoning_format}

Facts:
"""

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        """Enhanced completion formatting that handles reasoning steps"""
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
                    reasoning_patterns = tuple(f"{i}." for i in range(1, self.reasoning_steps + 1))
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

    def _generate_examples(self) -> str:
        """Generate dynamic examples based on reasoning steps"""
        if self.reasoning_steps == 1:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What verifiable medical facts can be extracted from this sentence? - None, this is a personal narrative about speaking to a doctor

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What verifiable medical facts can be extracted from this sentence? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 2:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 3:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 4:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience
4. Are these facts complete and contextually appropriate? - N/A, no verifiable facts

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects
4. Are these facts complete and contextually appropriate? - Yes, they are complete and contextually appropriate

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 5:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience
4. Are these facts complete and contextually appropriate? - N/A, no verifiable facts
5. Are these facts accurate and comprehensive? - N/A, no verifiable facts

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects
4. Are these facts complete and contextually appropriate? - Yes, they are complete and contextually appropriate
5. Are these facts accurate and comprehensive? - Yes, they accurately represent the medical information

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        else:
            # Default to 3 steps for any other value
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience

Facts:
- No verifiable claim

Context: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

    def _generate_reasoning_steps(self) -> str:
        """Generate dynamic reasoning steps based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. Identify and extract all verifiable medical facts from the sentence"
        elif self.reasoning_steps == 2:
            return """1. First, identify the main medical concepts in the sentence
2. Then, extract verifiable facts from each concept"""
        elif self.reasoning_steps == 3:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Finally, ensure each fact is objective and can be verified against reliable sources"""
        elif self.reasoning_steps == 4:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Next, ensure each fact is objective and can be verified against reliable sources
4. Finally, validate that each fact is complete and contextually appropriate"""
        elif self.reasoning_steps == 5:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Next, ensure each fact is objective and can be verified against reliable sources
4. Then, validate that each fact is complete and contextually appropriate
5. Finally, review and refine the extracted facts for accuracy and completeness"""
        else:
            # Default to 3 steps for any other value
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Finally, ensure each fact is objective and can be verified against reliable sources"""

    def _generate_reasoning_format(self) -> str:
        """Generate dynamic reasoning format based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. What verifiable medical facts can be extracted from this sentence?"
        elif self.reasoning_steps == 2:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?"""
        elif self.reasoning_steps == 3:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?"""
        elif self.reasoning_steps == 4:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?
4. Are these facts complete and contextually appropriate?"""
        elif self.reasoning_steps == 5:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?
4. Are these facts complete and contextually appropriate?
5. Are these facts accurate and comprehensive?"""
        else:
            # Default to 3 steps for any other value
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?"""
