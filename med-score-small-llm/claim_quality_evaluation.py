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
        self.llm.max_tokens = 4096
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
            d['other_claims'] = other_claims  # Store other claims for output

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
        prompt = f"""You are a meticulous medical information auditor. Your task is to classify a given "Atomic Claim" by comparing it against its "Context", "Original Sentence" and "Other Claims" list.
"Context" is the full response, "Original Sentence" is one sentence from that response, "Atomic Claim" is a specific claim derived from that sentence, and "Other Claims" list are all other claims derived from that same sentence (excluding the current claim being evaluated).

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
Analyze the "Atomic Claim" in isolation, but use the "Original Sentence" as the single source of truth for grounding and the "Context" to resolve ambiguities (like pronouns).

Step 1: Check for Unverifiable
Is the claim a personal narrative, a patient-specific experience, or an empathetic "bedside manner" statement? These claims cannot be verified by an external knowledge base.
- General Examples: "I spoke with your doctor.", "You are experiencing pain.", "Your pain can be very tiring.", "If you have any concerns, please don't hesitate to reach out to them."
- If YES: Label as Unverifiable. The process stops here.
- If NO: Proceed to Step 2.

Step 2: Check for Incorrectly structured
Is the claim a question, a command, or does it incorrectly include the reporting frame (e.g., "The doctor said...")? Or is the "Atomic Claim" only contain "Facts:" without any medical claims?
- General Examples: "Take ibuprofen for your pain.", "Did you take the medication?", "The study shows that aspirin is effective.", "Facts:".
- If YES: Label as Incorrectly structured. The process stops here.
- If NO: Proceed to Step 3.

Step 3: Check for Context-dependent
Does the claim contain unresolved pronouns (he, she, it, your, their) or vague references ("the medication," "the symptoms," "the condition") that make it impossible to understand without the original context?
- General Examples: "It may cause side effects.", "Your symptoms could be related to anovulatory cycles." (This is context-dependent unless "your symptoms" was replaced with "Irregular periods and extreme pain").
- If YES: Label as Context-dependent. The process stops here.
- If NO: Proceed to Step 4.

Step 4: Check for Incomplete
Does the claim DROP a critical medical modifier (like 'may', 'rarely'), condition (like 'if you have X')?
- CRITICAL RULE 1 (Reporting Frames): This rule does NOT apply to the removal of reporting frames (e.g., "The doctor believes that...", "The study shows that..."). Stripping these frames is a correct part of structuring the claim (which is checked in Step 2) and does NOT make the core medical fact incomplete.
- CRITICAL RULE 2 (Valid Decomposition): This rule does NOT apply to valid decomposition. If an "Original Sentence" contains multiple distinct facts (e.g., "A causes B and C" or "A and B cause C"), a claim that correctly and completely extracts just one of those facts (e.g., "A causes C" or "B causes C") is NOT Incomplete. It is a valid atomic fact.
- CRITICAL RULE 3 (Modifiers and Conditions):  ONLY focus on DROPPING critical medical modifier (like 'may', 'rarely'), condition (like 'if you have X'). If the claim changes the "Original Sentence" meaning in any other way (e.g., distorts, adds new info), that is checked in Step 5 (Hallucinated).
- General Incomplete Example 1 (Loses Modifier):
    - Original: "Anabolic steroids may have positive effects on muscle health."
    - Claim: "Anabolic steroids have positive effects on muscle health." (Loses "may" - this IS Incomplete).
- General Incomplete Example 2 (Loses Condition):
    - Original: "Growth hormones should only be taken if there is a diagnosed deficiency."
    - Claim: "Growth hormones should only be taken." (Loses "if there is a diagnosed deficiency" - this IS Incomplete).
- General NOT Incomplete Example 3 (Valid Decomposition):
    - Original: "A causes B and C."
    - Claim: "A causes C." (This is a complete atomic fact, not an incomplete one).
- General NOT Incomplete Example 4 (Frame Stripped):
    - Original: "The doctor believes that A causes B."
    - Claim: "A causes B." (This is a correctly structured, complete fact, not an incomplete one).
- General Example 5: (Hallucinated, not Incomplete)
    - Original: "10 days and a half may not be sufficient."
    - Claim: "10 days may not be sufficient." (This is NOT Incomplete - it is Hallucinated. Although the modifier "may" is retained, but the meaning is distorted by changing "10 days and a half" to "10 days". This is checked in Step 5).
- If YES (like Examples 1 & 2): Label as Incomplete. The process stops here.
- If NO (like Example 3 & 4 & 5): Proceed to Step 5.

Step 5: Check for Hallucinated
First, check if the "Atomic Claim" adds new medical information, distorts, or contradicts the "Original Sentence".
- If NO: The claim is grounded. Proceed to Step 6.
- If YES: The claim has information not in the "Original Sentence". You must now perform a "Context Check" to see if this is a valid substitution or a hallucination.
    - "Context Check": Is the new/changed information a direct and justifiable substitution for a pronoun (e.g., *it, they, your*) or a vague term (e.g., *the symptoms, the condition*) in the "Context"?
        - If YES (it's a valid substitution): The claim is NOT a hallucination. Proceed to Step 6.
            Note: The "Atomic Claim" considers Hallucinated if it is correct in "Context" but not in "Original Sentence".
        - If NO (it's new, unjustified info): The claim adds information that cannot be justified by either the "Original Sentence" or the "Context". Label as **Hallucinated**. The process stops here.
- General Hallucinated Example (New Info):
    - Context: "Aspirin may help reduce pain and inflammation."
    - Original Sentence: "Aspirin may help reduce pain."
    - Atomic Claim: "Aspirin, WHICH IS AN NSAID, may help reduce pain."
    - Reasoning: The phrase "which is an NSAID" is new information not in the "Original Sentence". You do the "Context Check", it is not a direct and justifiable substitution for a pronoun or a vague term in the "Context". This IS a hallucination.
- General Hallucinated Example (Distortion):
    - Original: "Aspirin may help reduce pain."
    - Claim: "Aspirin may help eliminate pain."
    - Reasoning: "Eliminate" distorts the meaning of "reduce". This IS a hallucination.
- General NOT Hallucinated Example (Valid Substitution):
    - Context: "...We were discussing Aspirin. It may help reduce pain..."
    - Original Sentence: "It may help reduce pain."
    - Claim: "Aspirin may help reduce pain."
    - Reasoning: The claim adds "Aspirin," which is not in the "Original Sentence." However, the "Context Check" confirms "Aspirin" is a valid substitution for the pronoun "It." This is NOT a hallucination.
- General Hallucinated Example (Correct in Context but not in Original Sentence):
    - Context: "The doctor mentioned that A is B. B may help reduce pain."
    - Original Sentence: "B may help reduce pain."
    - Claim: "A is B."
    - Reasoning: The claim "A is B" is correct in the "Context" but not present in the "Original Sentence." This IS a hallucination.

Step 6: Check for Redundant
This step requires the "Other Claims" list. Check these conditions IN ORDER.
Condition 1: Is it a Composite Claim?
Is the "Atomic Claim" a "composite claim" (e.g., "A and B") where its more atomic parts ("A", "B") are already present in the "Other Claims" list?
- General Example 1 (IS Redundant):
    - Claim to Evaluate: "A is B and C"
    - Other Claims List: ["A is B.", "A is C."]
    - Judgment: This claim meets Condition 1. It is a composite of claims already in the list.
- If YES: Label as Redundant. The process stops here.
- If NO: Proceed to check Condition 2.

Condition 2: Is it a Duplicate or Rephrasing?
Is the "Atomic Claim" a "direct duplicate" or "semantically identical rephrasing" of any claim in the "Other Claims" list?
- If NO: The claim is not redundant. Proceed to Step 7.
- If YES: You must perform a "completeness check" to decide the label.
    - 1. Find all claims in "Other Claims" that are duplicates/rephrasings of the "Atomic Claim".
    - 2. If the "Atomic Claim" is "more complete" (i.e., it contains more accurate detail or nuance) than ALL the other duplicate claims: Label as Valid. The process stops here.
    - 3. Otherwise, Label as Redundant. The process stops here.
    
CRITICAL RULE: If the "Atomic Claim" and the claims in "Other Claims" are all distinct, different atomic facts, they are NOT redundant. Do NOT misclassify two different facts as "rephrasings" just because they share a topic.
- Example (IS NOT Redundant - Atomic Part):
    - Claim to Evaluate: "A is B."
    - Other Claims List: ["A is B and C", "A is C."]
    - Judgment: This claim is NOT REDUNDANT. It is an atomic fact, not a composite. (It does not meet Condition 1 or 2).
- Example (IS NOT Redundant - Distinct Facts):
    - Other Claims List: ["A is likely a B."]
    - Claim to Evaluate: "B is a common occurrence."
    - Judgment: Although "Atomic Claim" and "Other Claims" share the same topic, this claim is NOT redundant. It is a completely different, distinct atomic fact. One fact is a diagnosis, the other is a definition. It does not meet Condition 1 (it's not a composite) or Condition 2 (it's not a rephrasing).

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
Step 1: Check for Unverifiable. The claim "I spoke to your doctor" describes a personal interaction or narrative. It cannot be externally verified. This matches. The process stops here.
Classification: Unverifiable
---
2. Example: Incorrectly structured
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "The doctor noted that these substances may have positive effects on muscle and bone health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative. Proceed.
Step 2: Check for Incorrectly structured. The claim incorrectly includes the reporting frame ("The doctor noted that..."). This matches. The process stops here.
Classification: Incorrectly structured

Original Sentence: "Fortunately, there are..."
Atomic Claim to Evaluate: "Facts:"
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative. Proceed.
Step 2: Check for Incorrectly structured. The claim only contains "Facts:" without any medical claims. This matches. The process stops here.
Classification: Incorrectly structured
---
3. Example: Context-dependent
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "They also carry significant risks."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative. Proceed.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. Proceed.
Step 3: Check for Context-dependent. The claim relies on the unresolved pronoun "They." Without the context, the claim is not standalone. This matches. The process stops here.
Classification: Context-dependent
---
4. Example: Incomplete
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids have positive effects on muscle health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative sentence. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Incomplete. The claim drops the critical modifier "may" from within the core medical fact. This is not a reporting frame. This matches General Incomplete Example 1. The process stops here.
Classification: Incomplete
---
5. Example: Hallucinated
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids may have negative effects on muscle health."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative sentence. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Incomplete. Retains all modifiers. Proceed.
Step 5: Check for Hallucinated. The original sentence uses "positive effect". The claim distorts this to "negative effect". This is a distortion of the original meaning. This matches. The process stops here.
Classification: Hallucinated
---
6. Example: Redundant
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids carry significant risks and potential side effects."
Other Claims: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids may have positive effects on bone health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Incomplete. Retains all modifiers. Proceed.
Step 5: Check for Hallucinated. The claim does not add new information. Grounded in the original. Proceed.
Step 6: Check for Redundant. 
Check Condition 1 (Composite): The claim being evaluated is a composite of 2 items in "Other Claims" including "Anabolic steroids carry significant risks." and "Anabolic steroids carry potential side effects.". This matches Condition 1. The process stops here.
Classification: Redundant

Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids may have positive effects on muscle health."
Other Claims: ["Anabolic steroids may have positive effects on muscle and bone health.", "Anabolic steroids may have positive effects on bone health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative. Proceed.
Step 3: Check for Context-dependent. Standalone. Proceed.
Step 4: Check for Incomplete. Retains all modifiers. Proceed.
Step 5: Check for Hallucinated. The claim does not add new information. Grounded in the original. Proceed.
Step 6: Check for Redundant. 
Check Condition 1 (Composite): The claim is atomic, not composite.
Check Condition 2 (Duplicate): The claim is one fact of "Anabolic steroids may have positive effects on muscle and bone health." in the "Other Claims" list, but it is atomic and enough medical detail. Moreover, "Anabolic steroids may have positive effects on muscle and bone health." in the "Other Claims" list is composite, not atomic and it is Redundant when evaluating it. Therefore, the claim is not a duplicate or rephrasing of any other claim in the list. The claim is not redundant. Proceed.
Step 7: Assign Valid. The claim has passed all six checks.
Classification: Valid
---
7. Example: Valid
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim to Evaluate: "Anabolic steroids may have positive effects on bone health."
Other Claims: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
Reasoning:
Step 1: Check for Unverifiable. Not a narrative. Proceed.
Step 2: Check for Incorrectly structured. Declarative. Proceed.
Step 3: Check for Context-dependent. Standalone ("Anabolic steroids" correctly replaces "these substances"). Proceed.
Step 4: Check for Incomplete. The claim retains the critical modifier "may". It is a validly decomposed atomic fact (like "CRITICAL RULE 2") from a larger sentence, not an incomplete one. Proceed.
Step 5: Check for Hallucinated. The claim is grounded. The term "Anabolic steroids" is a valid substitution for "these substances" from the "Context". It does not add new, un-grounded information. Proceed.
Step 6: Check for Redundant. 
Check Condition 1 (Composite): The claim is atomic, not composite. The "Other Claims" list contains other "distinct atomic facts". Proceed.
Check Condition 2 (Duplicate): The claim is not a duplicate or rephrasing of any other claim in the list (the other claims are distinct medical facts about muscle health, risks, and side effects). The claim is not redundant. Proceed.
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
