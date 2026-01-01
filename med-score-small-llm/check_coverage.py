import json
from typing import List


def format_coverage(context: str, sentence: str, current_claims: List[str]) -> str:
    formatted_current_claims = json.dumps(current_claims)
    prompt = f"""You are a strict Medical Information Coverage Auditor. Your goal is to determine if a list of "Current Claims" fully captures ALL medically significant information from an "Original Sentence".

INPUT DATA
1. Context: full medical response or passage (used ONLY to resolve pronouns and references).
2. Original Sentence: the specific source sentence in the Context that Current Claims are split from.
3. Current Claims: the list of atomic facts extracted from the Original Sentence.

---
INSTRUCTION
Compare the [Original Sentence] against the [Current Claims]. You must detect if any Medical Information is missing.

Rules:
1. IGNORE wording differences if the meaning is preserved (e.g., "PO" vs "by mouth" is ACCEPTABLE).
2. IGNORE non-medical narrative filler (e.g., "I told the patient", "It is important to note").

---
REASONING PROCESS
Step 1: CHECK IF CURRENT CLAIMS IS EMPTY
- If Current Claims is empty ([]):
    - Is it a personal narrative? (e.g., "I spoke to your doctor", "I told the patient")
    - Is it patient-specific experience? (e.g., "You are feeling pain")
    - Is it empathetic/bedside manner? (e.g., "It is understandable", "Your pain can be very tiring")
- If YES (Unverifiable) -> Coverage: FULL (no medical information to extract, empty claims is correct).
- If NO (has verifiable medical info) -> Coverage: MISSING (proceed to Step 2 to identify what's missing).

Step 2: ENTITY AUDIT
- Identify every medical entity in the Original Sentence.
- Verify: Are these entities present in the Current Claims list?
- If any entities are missing -> [missing entities].
- If all entities are present -> [].

Step 3: VERDICT
- If the list in Step 2 is not empty -> Result: {{"coverage": "MISSING", "missing": [list from Step 2]}}.
- If the list in Step 2 is empty -> Result: {{"coverage": "FULL", "missing": []}}.

---
FEW-SHOT EXAMPLES
Example 1:
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Original Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Current Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: CHECK IF CURRENT CLAIMS IS EMPTY
- Current Claims is not empty. Proceed to Step 2.
Step 2: ENTITY AUDIT
- Entities in Original Sentence: "these substances" (Anabolic steroids), "muscle health", "bone health", "risks", "side effects".
- Verify presence in Current Claims: "Anabolic steroids" is present in all claims. "muscle health" found in claim 1. "bone health" found in claim 2. "risks" found in claim 3. "side effects" found in claim 4.
- All entities are present. Missing entities: [].
Step 3: VERDICT
- The list in Step 2 is empty. -> Result: {{"coverage": "FULL", "missing": []}}.
Output:
{{
    "coverage": "FULL",
    "missing": []
}}

Example 2:
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Original Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Current Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids carry significant risks.",
]
Reasoning:
Step 1: CHECK IF CURRENT CLAIMS IS EMPTY
- Current Claims is not empty. Proceed to Step 2.
Step 2: ENTITY AUDIT
- Entities in Original Sentence: "these substances" (Anabolic steroids), "muscle health", "bone health", "risks", "side effects".
- Verify presence in Current Claims: "Anabolic steroids" (Present). "muscle health" (Present). "risks" (Present). "bone health" (MISSING). "side effects" (MISSING).
- Missing entities: ["bone health", "side effects"].
Step 3: VERDICT
The list in Step 2 is not empty -> Result: {{"coverage": "MISSING", "missing": ["bone health", "side effects"]}}.
Output:
{{
    "coverage": "MISSING",
    "missing": ["bone health", "side effects"]
}}

Example 3:
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Original Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Current Claims: []
Reasoning:
Step 1: CHECK IF CURRENT CLAIMS IS EMPTY
- Current Claims is empty.
- Is it a personal narrative? NO.
- Is it patient-specific experience? NO.
- Is it empathetic/bedside manner? NO.
- Result: NO (It has verifiable medical info).
- Coverage: MISSING. Proceed to Step 2 to identify what's missing.
Step 2: ENTITY AUDIT
- Entities in Original Sentence: "these substances" (Anabolic steroids), "muscle health", "bone health", "risks", "side effects".
- Verify presence in Current Claims: Current Claims is empty, so all entities are missing.
- Missing entities: ["muscle health", "bone health", "risks", "side effects"].
Step 3: VERDICT
The list in Step 2 is not empty -> Result: {{"coverage": "MISSING", "missing": ["muscle health", "bone health", "risks", "side effects"]}}.
Output:
{{
    "coverage": "MISSING",
    "missing": ["muscle health", "bone health", "risks", "side effects"]
}}

Example 4:
Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit.
Sentence: I spoke to your doctor and they wanted to address your concerns about tetanus.
Current Claims: []
Reasoning:
Step 1: CHECK IF CURRENT CLAIMS IS EMPTY
- Current Claims is empty.
- Is it a personal narrative? YES ("I spoke to your doctor").
- Result: YES (Unverifiable) -> Coverage: FULL (no medical information to extract, empty claims is correct).
Output:
{{
    "coverage": "FULL",
    "missing": []
}}

---
OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
Step 3: [Step 3 reasoning]
Output:
{{
    "coverage": "FULL" or "MISSING",
    "missing": [list of missing medical information, if any]
}}

---
YOUR TASK

Context: {context}
Original Sentence: {sentence}
Current Claims: {formatted_current_claims}
Output:"""
    return prompt


import ollama

context = """I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."""
sentence = """They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."""
claims = [
    "Irregular periods and extreme pain could be related to anovulatory cycles.",
    "Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.",
    "Irregular periods and extreme pain could be related to primary dysmenorrhea."
    "Primary dysmenorrhea is a condition that causes painful periods."
]

content = format_coverage(context=context, sentence=sentence, current_claims=claims)
print(content)

result = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{"role": "user", "content": content}],
    options={
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 4096,
    }
)

print(result)
with open('test_coverage_output.jsonl', 'w') as f:
    f.write(json.dumps({"context": context, "sentence": sentence, "claims": claims,
                        "response": result.message.content}) + '\n')
