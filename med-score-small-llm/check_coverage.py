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
Compare the "Original Sentence" against the "Current Claims". You must detect if any "Significant Medical Information" is missing.

Definition of "Significant Medical Information":
- Entities: Names of drugs, diseases, symptoms, body parts, procedures.
- Measurements: Dosages (mg, ml), Frequencies (twice daily), Durations (for 7 days), Vital signs numbers.
- Modifiers: Severity (severe, mild), Laterality (left, right), Negation (not, no, denies), Changes (worsening, improving).
- Conditions/Logic: Dependencies (if, when), Causality (due to), Sequence (before, after).

Rules:
1. IGNORE wording differences if the meaning is preserved (e.g., "PO" vs "by mouth" is coverage).
2. IGNORE non-medical narrative filler (e.g., "I told the patient", "It is important to note").
3. REPORT strictly what is missing. If nothing is missing, report coverage as "FULL".

---
REASONING PROCESS
Follow these 4 steps to determine coverage:

Step 1: ENTITY MAPPING
- Identify every medical entity in the Sentence.
- Check: Is this entity present in the Claims?

Step 2: DETAIL CHECK
- For each entity, checking its attributes in the Sentence vs Claims:
  - Is the Dosage/Frequency captured?
  - Is the Negation (not/no) captured?
  - Is the specific Modifier (Left/Right/Severe) captured?
- Example: Sentence says "Left knee", Claim says "Knee". -> Gaps found: "Left".

Step 3: LOGIC CHECK
- Are the connections preserved? (e.g., "A causes B").
- Are conditions preserved? (e.g., "Take X IF Y happens").

Step 4: LIST MISSING DETAILS
- Explicitly list out exactly what medical information is found in the Sentence but ABSENT in the Claims.
- Be specific (e.g., "The dosage '500mg' is missing", "The condition 'if fever occurs' is missing").
- If nothing is missing, write "None".

Step 5: VERDICT
- If the list in Step 4 is not empty -> Result: MISSING.
- If the list in Step 4 is "None" -> Result: FULL.

---
FEW-SHOT EXAMPLES

Example 1 (Full Coverage):
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Original Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Current Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: ENTITY MAPPING
- Entities in Sentence: "these substances" (Anabolic steroids), "muscle health", "bone health", "risks", "side effects".
- Check Claims: "Anabolic steroids" is present in all claims (Correctly resolved from "these substances"). "muscle health" is present in Claim 1. "bone health" is present in Claim 2. "risks" is present in Claim 3. "side effects" is present in Claim 4.
- Result: All entities mapped.
Step 2: DETAIL CHECK
- Check Modifiers/Attributes: "positive effects": Present in Claim 1 & 2. Probability "may have": Present in Claim 1 & 2. "significant" (for risks): Present in Claim 3. "potential" (for side effects): Present in Claim 4.
- Result: All critical modifiers captured.
Step 3: LOGIC CHECK
- The sentence uses a "while... also..." structure to list pros and cons. The claims correctly decompose this into atomic facts without losing the causal link that the steroids cause these effects.
- Result: Logic preserved.
Step 4: LIST MISSING DETAILS
- None
Step 5: VERDICT
- List is "None" -> Result: FULL.
Output:
{
  "coverage_status": "FULL",
  "missing_details": []
}

Example 2 (missing modifier):
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Original Sentence: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Current Claims: [
"Anabolic steroids have effects on muscle health.",
"Anabolic steroids have effects on bone health.",
"Anabolic steroids carry risks.",
"Anabolic steroids have side effects."
]
Reasoning:
Step 1: ENTITY MAPPING
- Entities in Sentence: "these substances" (Anabolic steroids), "muscle health", "bone health", "risks", "side effects".
- Check Claims: "Anabolic steroids" is present in all claims (Correctly resolved from "these substances"). "muscle health" is present in Claim 1. "bone health" is present in Claim 2. "risks" is present in Claim 3. "side effects" is present in Claim 4.
- Result: All entities mapped.
Step 2: DETAIL CHECK
- Check Modifiers/Attributes: "positive effects": MISSING (Claim 1 only says "have effects").
- Result: All critical modifiers captured.
Step 3: LOGIC CHECK
- The sentence uses a "while... also..." structure to list pros and cons. The claims correctly decompose this into atomic facts without losing the causal link that the steroids cause these effects.
- Result: Logic preserved.
Step 4: LIST MISSING DETAILS
- None
Step 5: VERDICT
- List is "None" -> Result: FULL.
Output:
{
  "coverage_status": "FULL",
  "missing_details": []
}










Example 2 (Missing Measurement):
Context: Post-op care.
Original Sentence: "Keep the wound dry and change the dressing twice daily."
Current Claims:
- "Keep the wound dry."
- "Change the dressing."
Reasoning:
- Entities: Wound, Dressing.
- Details: "Dry" (Found). "Twice daily" (MISSING in claims).
- Verdict: Frequency of dressing change is omitted.
Output:
{
  "coverage_status": "MISSING",
  "missing_details": ["Frequency 'twice daily' for changing dressing is missing."]
}

Example 3 (Missing Condition):
Context: Hypertension management.
Original Sentence: "Call 911 immediately if you experience severe chest pain or shortness of breath."
Current Claims:
- "Call 911 immediately."
- "Patient might experience chest pain."
- "Patient might experience shortness of breath."
Reasoning:
- Entities: 911, Chest pain, Shortness of breath.
- Logic: The sentence is a conditional instruction ("IF you experience"). The claims state the symptoms as possibilities but miss the *condition* triggering the call.
- Details: "Severe" modifier for chest pain is missing.
- Verdict: The conditional logic and severity modifier are missing.
Output:
{
  "coverage_status": "MISSING",
  "missing_details": [
    "The condition 'if you experience...' linking symptoms to calling 911 is missing.",
    "The modifier 'severe' for chest pain is missing."
  ]
}

Example 4 (Missing Item in List):
Context: Allergies.
Original Sentence: "Patient is allergic to Penicillin, Sulfa drugs, and Latex."
Current Claims:
- "Patient is allergic to Penicillin."
- "Patient is allergic to Latex."
Reasoning:
- Entities: Penicillin (Found), Latex (Found), Sulfa drugs (MISSING).
- Verdict: One item from the allergy list is omitted.
Output:
{
  "coverage_status": "MISSING",
  "missing_details": ["Allergy to 'Sulfa drugs' is missing."]
}

---
OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
...
Output:
{
  "coverage_status": "FULL" | "MISSING",
  "missing_details": ["Description of missing item 1", "Description of missing item 2"]
}

---
YOUR TASK

Context: {context}
Original Sentence: {sentence}
Current Claims: {formatted_current_claims}
Output:"""
    return prompt


import ollama



context = """I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."""
sentence = """The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."""
claims = [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects."
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
    f.write(json.dumps({"context": context, "sentence": sentence, "claim": claim, "error_label": error_label,
                        "response": result.message.content}) + '\n')
