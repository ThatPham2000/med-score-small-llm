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

        return f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning.

REASONING PROCESS:
{reasoning_steps_text}

The facts should be objective and verifiable against reliable external information such as Wikipedia and PubMed. All subjective personal experiences ("I was or someone did") and personal narratives (stating a past event) are not verifiable and should not be included in the fact list. Facts should be situated within conditions in the sentence. Suggestions (e.g. "I recommend or Your doctor suggest") and opinions (e.g. "I think") should be transformed into objective facts by removing subjective words and pronouns to only retain the core information that can be verified. Imperative instructions ("do something") should be transformed into declarative facts ("doing something is helpful for some conditions").

If there is an overly specific entity such as "Your partner" or vague references (pronouns, this or that) in the fact, replace it with a general phrase with conditional modifiers using information in the provided context (e.g. "People in some conditions"). Each fact should be verifiable on its own and require no additional context. Do not add additional information outside of the sentence and context.

REASONING FORMAT:
Think step by step:
{self._generate_reasoning_format()}

If there is no verifiable fact in the sentence, please write "No verifiable claim".

Here are some examples with reasoning:

Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. Main concepts: doctor consultation, tetanus concerns
2. Facts to extract: None - this is a personal narrative about speaking to a doctor
3. Verifiability: This is a subjective personal experience, not a verifiable medical fact

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. Main concepts: substances, muscle health, bone health, risks, side effects
2. Facts to extract: 
   - Substances can have positive effects on muscle health
   - Substances can have positive effects on bone health  
   - Substances carry risks
   - Substances carry side effects
3. Verifiability: These are objective medical facts about substance effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects.

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
                        if claim and claim != "No verifiable claim":
                            claim_list.append(claim)
                    elif line.strip() and not line.strip().startswith(("Context:", "Please breakdown")):
                        # Handle cases where facts don't start with bullet points
                        claim = line.strip()
                        if claim and claim != "No verifiable claim":
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
