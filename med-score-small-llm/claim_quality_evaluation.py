import asyncio
import json
from typing import List, Dict, Any

import nest_asyncio
from tqdm import tqdm

from llm import LLM
from utils import chunker

nest_asyncio.apply()


class ClaimQualityEvaluation(object):
    def __init__(
            self,
            llm: LLM = None,
            random_state: int = 42,
            batch_size: int = 32,
    ):
        self.llm = llm
        self.llm.max_tokens = 2048
        self.random_state = random_state
        self.batch_size = batch_size

    def do_claim_quality_evaluation(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Group decompositions by id (same response)
        decompositions_by_id = {}
        for d in decompositions:
            response_id = d['id']
            if response_id not in decompositions_by_id:
                decompositions_by_id[response_id] = []
            decompositions_by_id[response_id].append(d)

        messages = []
        for d in decompositions:
            # Get all other claims from the same response (same id) and same sentence, excluding current claim
            response_id = d['id']
            sentence = d.get('sentence', '')
            other_claims = [
                other_d['claim']
                for other_d in decompositions_by_id[response_id] if
                other_d['claim_id'] != d['claim_id'] and other_d.get('claim') is not None and other_d.get('sentence',
                                                                                                          '') == sentence
            ]

            formatted_input = self.format_input(d['context'], d['sentence'], d['claim'], other_claims)
            messages.append([
                {"role": "user", "content": formatted_input}
            ])

        all_completions = []
        n_iter = len(messages) // self.batch_size
        for batch in tqdm(chunker(messages, self.batch_size), desc="Claim quality evaluation process", total=n_iter):
            completions = asyncio.run(self.llm.batch_response(batch))
            all_completions.extend(completions)

        claim_quality_output = []
        for decomposition, completion in zip(decompositions, self.llm.normalize_llm_response(all_completions)):
            raw_output = completion.strip()
            claim_quality_type = self.parse_claim_quality_output(raw_output)
            output = {k: v for k, v in decomposition.items()}
            output["raw_claim_quality_response"] = raw_output
            output["claim_quality_type"] = claim_quality_type
            claim_quality_output.append(output)
        return claim_quality_output

    def parse_claim_quality_output(self, completion_message: str) -> str:
        valid_types = [
            "Valid",
            "Unverifiable",
            "Incorrectly structured",
            "Context-dependent",
            "Hallucinated",
            "Incomplete",
            "Redundant"
        ]

        completion_message = completion_message.strip()

        # Try to find the classification label after "Classification:"
        classification_patterns = [
            "Classification:",
            "classification:",
            "Classification :",
            "classification :"
        ]

        classification_label = None

        # Search for classification label
        for pattern in classification_patterns:
            if pattern in completion_message:
                # Extract text after the pattern
                parts = completion_message.split(pattern, 1)
                if len(parts) > 1:
                    label_text = parts[1].strip()
                    # Extract the first word or phrase (handle multi-word labels)
                    # Remove any trailing punctuation or newlines
                    label_text = label_text.split('\n')[0].split('.')[0].strip()

                    # Try exact match first (case-insensitive)
                    for valid_type in valid_types:
                        if label_text.lower() == valid_type.lower():
                            classification_label = valid_type
                            break

                    # If no exact match, try partial match
                    if classification_label is None:
                        label_lower = label_text.lower()
                        for valid_type in valid_types:
                            if valid_type.lower() in label_lower or label_lower in valid_type.lower():
                                classification_label = valid_type
                                break

                    if classification_label:
                        break

        # Fallback: search for any valid type in the entire message (case-insensitive)
        if classification_label is None:
            message_lower = completion_message.lower()
            for valid_type in valid_types:
                if valid_type.lower() in message_lower:
                    # Check if it appears near the end (more likely to be the classification)
                    last_occurrence = message_lower.rfind(valid_type.lower())
                    if last_occurrence > len(completion_message) * 0.5:  # In the latter half
                        classification_label = valid_type
                        break

        # Default fallback
        if classification_label is None:
            classification_label = "Parse Failed"

        return classification_label

    def format_input(self, context: str, sentence: str, claim: str, other_claims: List[str]) -> str:
        formatted_other_claims = json.dumps(other_claims)
        prompt = f"""You are a meticulous medical information auditor. Your task is to classify a given "Atomic Claim" by comparing it against its "Context", "Original Sentence" and any "Other Claims" generated from that same sentence.

Your goal is to assign ONE of the following seven labels to the claim:
1. Valid
2. Unverifiable
3. Incorrectly structured
4. Context-dependent
5. Hallucinated
6. Incomplete
7. Redundant

GUIDING PRINCIPLE:
To ensure accuracy, you MUST evaluate the "Atomic Claim" by following these steps IN ORDER. The first category that matches is the correct classification.

Step 1: Check for Unverifiable
Is the claim a personal narrative, a patient-specific experience, or an empathetic "bedside manner" statement? These claims cannot be verified by an external knowledge base.
- General Examples: "I spoke with your doctor.", "You are experiencing pain.", "Your pain can be very tiring."
- If YES: Label as Unverifiable.
- If NO: Proceed to Step 2.

Step 2: Check for Incorrectly structured
Is the claim a question, a command, or does it incorrectly include the reporting frame (e.g., "The doctor said...")?
- General Examples: "Take ibuprofen for your pain.", "Did you take the medication?", "The study shows that aspirin is effective."
- If YES: Label as Incorrectly structured.
- If NO: Proceed to Step 3.

Step 3: Check for Context-dependent
Does the claim contain unresolved pronouns (he, she, it, your, their) or vague references ("the medication," "the symptoms," "the condition") that make it impossible to understand without the original context?
- General Examples: "It may cause side effects.", "Your symptoms could be related to anovulatory cycles." (This is context-dependent unless "your symptoms" was replaced with "Irregular periods and extreme pain").
- If YES: Label as Context-dependent.
- If NO: Proceed to Step 4.

Step 4: Check for Hallucinated
Does the claim add ANY new information (even a single word) that was not in the "Original Sentence"? Does it distort or contradict the original meaning?
    - General Original: "Aspirin may help reduce pain."
- General Hallucinated Example: "Aspirin, WHICH IS AN NSAID, may help reduce pain." (Adds "which is an NSAID").
- If YES: Label as Hallucinated.
- If NO: Proceed to Step 5.

Step 5: Check for Incomplete
Does the claim DROP a critical medical modifier, condition, or nuance from the "Original Sentence," thereby changing its meaning? This is the opposite of a hallucination.
- General Original: "Anabolic steroids may have positive effects on muscle health."
- General Incomplete Example: "Anabolic steroids have positive effects on muscle health." (Loses the critical modifier "may").
- General Original 2: "Growth hormones should only be taken if there is a diagnosed deficiency."
- General Incomplete Example 2: "Growth hormones should only be taken." (Loses the condition "if there is a diagnosed deficiency").
- If YES: Label as Incomplete.
- If NO: Proceed to Step 6.

Step 6: Check for Redundant
(This step requires the "Other Claims" list). Is this "Atomic Claim" a minimal rephrasing, a direct duplicate, or a composite of OTHER claims in the provided list?
- Example 1 (Redundant):
    - Other Claims List: ["Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
    - Claim to Evaluate: "Anabolic steroids carry significant risks and potential side effects."
    - Judgment: This claim is a composite of the other two more-atomic claims and is therefore REDUNDANT.
- Example 2 (NOT Redundant):
    - Other Claims List: ["Anabolic steroids carry significant risks and potential side effects.", "Anabolic steroids carry potential side effects."]
    - Claim to Evaluate: "Anabolic steroids carry significant risks."
    - Judgment: This claim is atomic. It is not a composite of other claims, so it is NOT redundant.
- If YES (like Example 1): Label as Redundant.
- If NO (like Example 2): Proceed to Step 7.

Step 7: Assign Valid
If the claim has passed all six previous checks, it is a Valid atomic fact. It is standalone, declarative, grounded, and complete.
- Label as Valid.

CLASSIFICATION EXAMPLES (FEW-SHOT LEARNING)

All the following examples are based on this Context and Original Sentence:

Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."

1. Example: Unverifiable
Original Sentence: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids..."
Atomic Claim to Evaluate: "I spoke to your doctor."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim "I spoke to your doctor" describes a personal interaction or narrative. It cannot be externally verified. This matches.
Classification: Unverifiable
---
2. Example: Incorrectly structured
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "The doctor noted that these substances may have positive effects on muscle and bone health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative. Proceed.
Step 2: Check for Incorrectly structured. The claim incorrectly includes the reporting frame ("The doctor noted that..."). This matches.
Classification: Incorrectly structured
---
3. Example: Context-dependent
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "They also carry significant risks."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative. Proceed.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. Proceed.
Step 3: Check for Context-dependent. The claim relies on the unresolved pronoun "They." Without the context, the claim is not standalone. This matches.
Classification: Context-dependent
---
4. Example: Hallucinated
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids may have negative effects on muscle health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative sentence. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Hallucinated. The original sentence uses "positive effect". The claim distorts this to "negative effect". This is a distortion of the original meaning. This matches.
Classification: Hallucinated
---
5. Example: Incomplete
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids have positive effects on muscle health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative sentence. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Hallucinated. The claim does not add new information. Proceed.
Step 5: Check for Incomplete. The original sentence contains the critical modifier "may". The claim omits "may", changing the meaning from a possibility to a definite fact. This matches.
Classification: Incomplete

6. Example: Redundant
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids carry significant risks and potential side effects."
Other Claims: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids may have positive effects on bone health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Hallucinated. Grounded in the original. Proceed.
Step 5: Check for Incomplete. Retains all modifiers. Proceed.
Step 6: Check for Redundant. The "Other Claims" list includes "Anabolic steroids carry significant risks." and "Anabolic steroids carry potential side effects." The claim being evaluated is a composite of these two other, more atomic claims. This matches.
Classification: Redundant
---
7. Example: Valid
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids may have positive effects on bone health."
Other Claims: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative. Proceed.
Step 3: Check for Context-dependent. Standalone ("Anabolic steroids" correctly replaces "these substances"). Proceed.
Step 4: Check for Hallucinated. Grounded in the original ("may have positive effects on bone health"). Proceed.
Step 5: Check for Incomplete. Retains the critical modifier "may". Proceed.
Step 6: Check for Redundant. The claim is atomic and distinct from the "Other Claims" provided. Proceed.
Step 7: Assign Valid. The claim has passed all six checks.
Classification: Valid
---
OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
...
Classification: (Your single-word classification label)
---
YOUR TASK

You will be given the context, the original sentence, the specific claim to evaluate, and the other claims from that sentence. You must follow the reasoning process from the examples.

Context: {context}
Original Sentence: {sentence}
Atomic Claim to Evaluate: {claim}
Other Claims: {formatted_other_claims}"""
        return prompt
