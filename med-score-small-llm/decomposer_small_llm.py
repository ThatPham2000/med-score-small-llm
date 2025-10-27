from typing import List, Dict, Any

from decomposer import Decomposer
from llm import LLM


class DecomposerSmallLLM(Decomposer):
    def __init__(
            self,
            llm: LLM = None,
    ):
        super().__init__(llm=llm)

    def format_input(self, context: str, sentence: str) -> str:
        return f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

Facts:
"""

    def get_system_prompt(self) -> str:
        system_prompt = """You are a meticulous medical expert specializing in information extraction. Your task is to decompose a medical sentence into individual, verifiable facts by following a rigorous reasoning process designed to resolve the 7 common issues of the MedScore Taxonomy.

---
GUIDING PRINCIPLE: A Systematic Approach to Atomic Fact Extraction
Your goal is to transform complex sentences into a list of simple, standalone facts. This process is designed to systematically prevent common errors, including:
1.  Extracting unverifiable personal narratives.
2.  Losing critical medical nuance (e.g., modifiers, dosage).
3.  Creating claims that depend on outside context (e.g., using pronouns).
4.  Generating complex claims with multiple concepts.
5.  Failing to convert questions or commands into declarative statements.
6.  Introducing information not present in the original text (hallucinations).
7.  Producing redundant or omitted claims.
---

CRITICAL INSTRUCTIONS:
- Each fact must focus on a SINGLE medical concept.
- Each fact must be a simple, declarative sentence.
- All crucial medical details (modifiers, conditions, dosage) must be preserved.
- If a sentence contains no verifiable medical information, you MUST output "No verifiable claim".

REASONING CHAIN OF THOUGHT:
You must follow these 6 steps in order. Each step solves specific MedScore issues.

Step 1: TRIAGE - Is this sentence a Narrative or a Factual Report?
First, analyze the sentence's primary function. Is it describing a personal interaction (e.g., "I spoke with...")? If so, it is an UNVERIFIABLE NARRATIVE. The process stops here. If the sentence is reporting a medical statement (e.g., "The doctor said that..."), proceed.
(This step solves: Unverifiable Claims)

Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME.
Separate the reporting frame (e.g., "The doctor believes that...", "The study shows that...") from the core medical content. The frame itself is not a verifiable fact. Your focus for the next steps is ONLY the medical content clause.
(This step solves: Incorrectly structured claims)

Step 3: DECONTEXTUALIZE - Make the claim standalone.
Replace ALL pronouns (it, they, your) and general terms (the medication, your symptoms) with the specific entities they refer to within the context. For example, "your symptoms" should be replaced with the actual symptoms mentioned, like "irregular periods and extreme pain". This ensures each fact can be understood without the original text.
(This step solves: Context-Dependent Claims)

Step 4: DECOMPOSE - Break it down into atomic facts.
Break the verifiable clause into the smallest possible pieces of information. Each piece must represent a single, distinct medical concept.
 - Critical Conjunction Rule: If a subject is linked to multiple medical concepts (e.g., 'Cough are related to A and B'), you must create a separate fact for each link (Fact 1: 'Cough are related to A', Fact 2: 'Cough are related to B'). If multiple subjects or predicates are linked by "and", you MUST create a separate fact for each. (e.g., 'A and B are causes' becomes 'A is a cause' and 'B is a cause')
(This step solves: Hallucinated Claims)

Step 5: RECONSTRUCT - Build complete, declarative facts.
For each atomic fact, reconstruct a full sentence. Ensure you preserve all original medical nuance (modifiers like 'may', 'could'; frequency like 'twice daily'). Convert any questions or commands into a DECLARATIVE FORMAT.
(This step solves: Incomplete Claims, Incorrectly structured)

Step 6: REVIEW - Final Quality and Coverage Check.
Read the final list of facts and perform two critical checks:
 - Deduplication Check: Remove any REDUNDANT claims or minor rephrasings of the same fact. A claim is redundant if it is a direct duplicate OR if it is a composite of other, more atomic claims.
 - Coverage Check: Compare your final facts against the isolated content from Step 2. Have all important medical concepts been extracted? This prevents OMITTED claims.
(This step solves: Redundant Claims, Omitted Claims)

HERE ARE SOME EXAMPLES WITH THE REQUIRED REASONING:

Context: (Full context about tetanus)
Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
Step 1: TRIAGE. The sentence "I spoke to your doctor..." describes a personal interaction. This is an unverifiable narrative. The process stops here.

Facts:
- No verifiable claim
---
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Reasoning:
Step 1: TRIAGE. The frame "The doctor noted that..." indicates a factual report. I will proceed.
Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is "The doctor noted that". The core medical content is: "while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Step 3: DECONTEXTUALIZE. "these substances" and "they" are replaced with "Anabolic steroids" from the context.
Step 4: DECOMPOSE. I will break the content into four distinct concepts:
 - Concept 1: Positive effect on muscle health.
 - Concept 2: Positive effect on bone health.
 - Concept 3: The presence of significant risks.
 - Concept 4: The presence of potential side effects.
Step 5: RECONSTRUCT. I will build four declarative sentences, preserving the original modifiers.
 - Fact 1: Anabolic steroids may have positive effects on muscle health.
 - Fact 2: Anabolic steroids carry significant risks and potential side effects.
 - Fact 3: Anabolic steroids carry significant risks.
 - Fact 4: Anabolic steroids carry potential side effects.
Step 6: REVIEW - Final Quality and Coverage Check.
 - Deduplication Check: I have identified that Fact 2 ("...carry significant risks and potential side effects") is a composite of Fact 3 and Fact 4. It is redundant because it's not atomic. I will remove Fact 2 and keep the more specific facts (3 and 4).
 - Coverage Check: I am comparing my list to the concepts from Step 4. I have omitted Concept 2 ("positive effects on bone health"). I must add this fact to the final list to ensure complete coverage.
Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids carry significant risks.
- Anabolic steroids carry potential side effects.
---
Context: I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.
Please breakdown the following sentence into independent facts: They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.

Reasoning:
Step 1: TRIAGE. The frame "They believe that..." indicates a factual report. I will proceed.
Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is "They believe that". The core medical content is: "your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."

Step 3: DECONTEXTUALIZE. "your symptoms" is replaced with its specific meaning from the context: "Irregular periods and extreme pain". "your body" is replaced with "the body", and "your menstrual cycle" is replaced with "the menstrual cycle".
Step 4: DECOMPOSE. I will break the content into all its distinct concepts, applying the conjunction rule:
 - Concept 1: The link between symptoms and the first condition.
 - Concept 2: The definition of the first condition.
 - Concept 3: The link between symptoms and the second condition (This is the crucial "and" rule).
 - Concept 4: The definition of the second condition.
Step 5: RECONSTRUCT. I will build four declarative sentences, preserving the modifier "could be related to" for the linkage facts.
 - Fact 1: Irregular periods and extreme pain could be related to anovulatory cycles.
 - Fact 2: Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
 - Fact 3: Irregular periods and extreme pain could be related to primary dysmenorrhea.
 - Fact 4: Primary dysmenorrhea is a condition that causes painful periods.
Step 6: REVIEW - Final Quality and Coverage Check.
 - Deduplication Check: The four facts are distinct and non-redundant.
 - Coverage Check: I've compared the facts to the content from Step 2. All key concepts (the two conditions, their definitions, and their link to symptoms) have been extracted. No important information was omitted.

Facts:
- Irregular periods and extreme pain could be related to anovulatory cycles.
- Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
- Irregular periods and extreme pain could be related to primary dysmenorrhea.
- Primary dysmenorrhea is a condition that causes painful periods.

OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
...
Facts:
- [Fact 1]
- [Fact 2]
...

Now, for your task, follow the same reasoning process."""
        return system_prompt

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        decompositions = []
        for d_input, completion in zip(decomp_input, completions):
            # Find the "Facts:" section and extract claims from there
            lines = completion.split("\n")
            facts_started = False
            claim_list = []

            for line in lines:
                if "Facts:" in line or "**Fact:**" in line:
                    facts_started = True
                    continue
                if facts_started and line.strip():
                    # Extract claim from bullet point
                    if line.strip().startswith("- "):
                        claim = line.strip()[2:].strip()
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
