import json
from typing import List


def format_input(context: str, sentence: str, claim: str, other_claims: List[str]) -> str:
    formatted_other_claims = json.dumps(other_claims)
    prompt = f"""You are a Medical Information Auditor. Your task is to evaluate the quality of a specific "Atomic Claim" extracted from a medical sentence.
    
INPUT DATA:
- "Context": The full response (used for resolving pronouns/references).
- "Original Sentence": The specific source sentence in the Context.
- "Atomic Claim": The specific fact you need to classify.
- "Other Claims": A list of other facts extracted from the same sentence.

---
INSTRUCTIONS:
Classify the "Atomic Claim" into exactly ONE of the following 7 categories.
You must evaluate the claim against the rules below in order (Step 1 to Step 7). The first step that matches determines the label.

The 7 Labels:
1. Unverifiable: Personal narratives or empathy.
2. Incorrectly structured: Questions, commands, or contains reporting frames.
3. Context-dependent: Contains unresolved pronouns or vague terms.
4. Incomplete: Missing critical modifiers (may, severe) or conditions (if, when).
5. Hallucinated: Adds info not in "Atomic Claim" or distorts meaning.
6. Redundant: A duplicate or a composite of other claims.
7. Valid: A perfect, standalone atomic fact.

---
REASONING PROCESS:
Step 1: Check for Unverifiable
Is the claim a personal narrative ("I spoke to..."), a patient-specific experience ("You are feeling..."), or an empathetic "bedside manner" statement ("It is understandable...")? These claims cannot be verified by an external knowledge base.
- IF YES -> STOP. Label: Unverifiable
- IF NO -> Proceed to Step 2.
Step 2: Check for Incorrectly structured
Does the claim:
a) Retain the reporting frame (e.g., "The doctor noted that...", "It is believed that...")?
b) Exist as a Question ("Did you take it?") or Command ("Take this.")?
c) Contain non-factual text (e.g., "Facts:", "Bullet point")?
- IF YES -> STOP. Label: Incorrectly structured
- IF NO -> Proceed to Step 3.
Step 3: Check for Context-dependent
Does the claim contain words that refer to something outside the claim itself?
- Look for Pronouns: he, she, it, they, this, these.
- Look for possessive adjectives (their, your, its, his, her).
- Is the sentence used in the first person? (I, we, my, our)
- Look for Vague References: the medication, the symptoms, the condition, the patient.
- Can a stranger understand exactly what this claim is about without reading the Context?
- IF NO (It needs context) -> STOP. Label: Context-dependent
- IF YES (It is specific) -> Proceed to Step 4.
Step 4: Check for Incomplete
Compare the Claim to the Original Sentence. Did the claim drop a Critical Medical Modifier?
- Critical Modifiers: Probability (may, could, likely), Frequency (daily, rarely), Severity (severe, mild), Conditions (if, when).
- Crucial Exception (Valid Decomposition): Do NOT flag as Incomplete if the claim splits a list "AND/OR" (e.g., "A causes B and C" -> "A causes B"). This is valid. Only flag if the meaning of the individual fact has changed due to a missing word.
- IF YES (Modifier/Condition missing) -> STOP. Label: Incomplete
- IF NO -> Proceed to Step 5.
Step 5: Check for Hallucinated
Does the claim assert something NOT present or implied in the Original Sentence?
- Check 1 (New Info): Does it add details (e.g., "Aspirin" becomes "Aspirin (NSAID)")? -> If the Context explicitly supports this substitution, it is VALID. If not, it is HALLUCINATED.
- Check 2 (Distortion): Does it change the meaning (e.g., "reduces pain" -> "eliminates pain")?
- IF YES -> STOP. Label: Hallucinated
- IF NO -> Proceed to Step 6.
Step 6: Check for Redundant
Look at the "Other Claims" list.
- Condition A (Composite): Is the Atomic Claim just a combination of two facts already in the list? (e.g., Claim="A and B", List=["A", "B"]).
- Condition B (Subset): Is the Atomic Claim a shorter/less detailed version of another claim in the list? (e.g., Claim="Pain", List=["Severe Pain"]).
- IF YES to A or B -> STOP. Label: Redundant
- IF NO -> Proceed to Step 7.
Step 7: Assign Valid
If the claim passed all checks above, Label is Valid.

---
FEW-SHOT EXAMPLES:
Example 1:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease."
Atomic Claim: "I spoke to your doctor."
Other Claims: []
Reasoning:
Step 1: Check for Unverifiable. The Atomic Claim "I spoke to your doctor" describes a specific personal interaction (event narrative) between the speaker and the doctor. It is not a verifiable general medical fact. STOP. Label: Unverifiable
Classification: Unverifiable

Example 2:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "The doctor noted that these substances may have positive effects on muscle and bone health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim contains medical content regarding the effects of substances, so it is not a purely personal narrative or empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim explicitly retains the reporting frame "The doctor noted that...", which should have been removed to make the claim a standalone fact. This matches the condition for incorrectly structured claims. STOP. Label: Incorrectly structured
Classification: Incorrectly structured

Example 3:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "They also carry significant risks."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.",
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim "They also carry significant risks" is a statement regarding medical risk, not a personal narrative or an empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence and does not include the reporting frame "The doctor noted that". Proceed to Step 3.
Step 3: Check for Context-dependent. The claim begins with the pronoun "They". Without reading the Context or Original Sentence, it is impossible to know what "They" refers to (which is "anabolic steroids"). Therefore, the claim relies on outside context. STOP. Label: Context-dependent
Classification: Context-dependent

Example 4:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids have positive effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim is not a personal narrative or empathy statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. It is a declarative sentence and does not contain the reporting frame "The doctor noted". Proceed to Step 3.
Step 3: Check for Context-dependent. The vague term "these substances" has been correctly resolved to "Anabolic steroids". It is standalone. Proceed to Step 4.
Step 4: Check for Incomplete. The Original Sentence states "these substances may have positive effects". The Atomic Claim states "Anabolic steroids have positive effects". The claim drops the critical probability modifier "may", which changes the meaning from a possibility to a certainty. This matches the Incomplete criteria. STOP. Label: Incomplete 
Classification: Incomplete

Example 5:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have negative effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on bone health.",
"Anabolic steroids carry significant risks.",
"Anabolic steroids carry potential side effects"
]
Reasoning:
Step 1: Check for Unverifiable. The claim "Anabolic steroids may have negative effects on muscle health" is a general medical statement, not a personal narrative or empathy. Proceed to Step 2.
Step 2: Check for Incorrectly structured. It is a declarative sentence, does not contain a reporting frame ("The doctor noted that..."), and is not a question or command. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses "Anabolic steroids" (specific entity) instead of "these substances". It stands alone clearly. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the modifier "may". Proceed to Step 5.
Step 5: Check for Hallucinated. Compare to the Original Sentence: "these substances may have positive effects on muscle...". The claim states "may have negative effects on muscle health." This contradicts and distorts the meaning of the original sentence regarding muscle health. This matches Check 2 (Distortion). Stop. Label: Hallucinated
Classification: Hallucinated

Example 6:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids carry significant risks and potential side effects."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.", 
"Anabolic steroids may have positive effects on bone health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim "Anabolic steroids carry significant risks and potential side effects" is a factual medical statement. It is not a personal narrative ("I spoke..."), patient experience ("You feel..."), or empathy statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. It does not contain a reporting frame (e.g., "The doctor noted that..."). It is not a question or command. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses the specific entity "Anabolic steroids" instead of the pronoun "they" or "these substances". It contains no vague terms requiring external context. It is specific and standalone. Proceed to Step 4.
Step 4: Check for Incomplete. Comparing to the original sentence segment "...they also carry significant risks and potential side effects", the claim preserves the modifiers "significant" and "potential". No critical conditions or modifiers are dropped. Proceed to Step 5.
Step 5: Check for Hallucinated. The claim is strictly grounded in the original text. It does not add new information or distort the meaning. Proceed to Step 6.
Step 6: Check for Redundant. Looking at the "Other Claims" list: ["Anabolic steroids may have positive effects on muscle health.", "Anabolic steroids may have positive effects on bone health.", "Anabolic steroids carry significant risks.", "Anabolic steroids carry potential side effects."]
- Condition A (Composite Check): The Atomic Claim "Anabolic steroids carry significant risks and potential side effects" is a direct combination of two facts already present in the list: "Anabolic steroids carry significant risks" AND "Anabolic steroids carry potential side effects". Since the more atomic versions exist in the list, this composite claim is Redundant. STOP. Label: Redundant.
Classification: Redundant

Example 7:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have positive effects on muscle health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle and bone health.",
"Anabolic steroids may have positive effects on bone health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim is a factual statement about the effects of a substance, not a personal narrative or empathy. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim is a declarative sentence. It does not contain the reporting frame ("The doctor noted that..."). Proceed to Step 3.
Step 3: Check for Context-dependent. The claim uses "Anabolic steroids" which correctly replaces the pronoun/reference "these substances" found in the Original Sentence based on the Context. It is specific and understandable to a stranger. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the critical modifier "may". It separates "muscle health" from "muscle and bone health", which is a Valid Decomposition (Crucial Exception) of an "AND" list. It is a complete atomic fact. Proceed to Step 5.
Step 5: Check for Hallucinated. The substitution of "Anabolic steroids" for "these substances" is explicitly supported by the Context ("...safety of using anabolic steroids..."). It does not add new unsupported info. Proceed to Step 6.
Step 6: Check for Redundant.
Condition A (Composite Check): The claim is atomic, not a combination of others.
Condition B (Subset Check): The claim is "Anabolic steroids may have positive effects on muscle health". The Other Claims list contains the composite "Anabolic steroids may have positive effects on muscle and bone health". However, the atomic claim is a distinct, validly separated concept, not a "less detailed/vague" version of the composite. It is not redundant to the other atomic claim ("bone health"). Proceed to Step 7.
Step 7: Assign Valid. The claim has passed all six checks.
Classification: Valid

Example 8:
Context: "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Original Sentence: "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Atomic Claim: "Anabolic steroids may have positive effects on bone health."
Other Claims: [
"Anabolic steroids may have positive effects on muscle health.", 
"Anabolic steroids carry significant risks.", 
"Anabolic steroids carry potential side effects."
]
Reasoning:
Step 1: Check for Unverifiable. The claim is a medical statement regarding health effects, not a personal narrative or empathetic statement. Proceed to Step 2.
Step 2: Check for Incorrectly structured. The claim properly removes the reporting frame ("The doctor noted that...") and is a complete declarative sentence. Proceed to Step 3.
Step 3: Check for Context-dependent. The claim resolves the vague phrase "these substances" from the Original Sentence to the specific entity "Anabolic steroids" based on the Context. It is specific and standalone. Proceed to Step 4.
Step 4: Check for Incomplete. The claim retains the critical probability modifier "may". Although the original sentence mentions "muscle and bone health", extracting "bone health" as a separate fact is a Valid Decomposition (splitting an "AND" list) and is not considered Incomplete. Proceed to Step 5.
Step 5: Check for Hallucinated. The claim is grounded in the text. The substitution of "Anabolic steroids" is explicitly supported by the Context. Proceed to Step 6.
Step 6: Check for Redundant. The claim focuses on "bone health". The Other Claims list focuses on "muscle health", "risks", and "side effects". The claim is distinct and is neither a composite nor a subset of the other claims. Proceed to step 7.
Step 7: Assign Valid. The claim passed all checks.
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

Context: {context}
Original Sentence: {sentence}
Atomic Claim: {claim}
Other Claims: {formatted_other_claims}"""
    return prompt


import ollama

# contentMap = {"id":"zxkyix_20241101","sentence_id":6,"sentence":"It's essential to keep in mind that lifestyle factors, such as smoking and obesity, are generally more significant risk factors for cancer.","claim":"Obesity is a risk factor for cancer.","claim_id":3,"model_response":"Facts:\n- Lifestyle factors are risk factors for cancer.\n- Smoking is a risk factor for cancer.\n- Obesity is a risk factor for cancer.","context":"I spoke to your doctor, and they expressed concern about the high number of scans you've had in the past year. They noted that having a chest X-ray more than once a month is unusual unless there's evidence of an acute change in your condition.\n\nRegarding your question about radiation, your doctor wants to reassure you that the radiation from chest X-rays is relatively negligible. However, CT scans do involve more radiation, and the exact dose depends on various factors, such as the area being scanned.\n\nYour doctor also wants to clarify that there's no specific \"tipping point\" where radiation suddenly becomes a significant risk. Instead, each scan increases your relative risk of cancer by a small amount, similar to how smoking increases your risk of cancer the more you smoke.\n\nIt's essential to keep in mind that lifestyle factors, such as smoking and obesity, are generally more significant risk factors for cancer. Your doctor recommends discussing your concerns with your radiologist or primary care physician to determine the best course of action for your specific situation.\n\nPlease let us know if you have any further questions or concerns.","other_claims":["Facts:","Lifestyle factors are risk factors for cancer.","Smoking is a risk factor for cancer."],"raw_claim_quality_response":"","claim_quality_type":"Parse Failed","manual_claim_quality_type":"Valid"}
contentMap = {"id": "4d0x3s_20241101", "sentence_id": 6, "sentence": "They will likely need to assess your knee and review your medical history to determine the best course of action.", "claim": "Reviewing medical history may be necessary to determine the best course of action.", "claim_id": 1, "context": "Dear Cedric,\n\nI spoke to your doctor, and they would like to know if you had a full range of motion in your knee at any point after your surgery in 2013. This information will help them better understand your current situation.\n\nAccording to your doctor, regaining full range of motion may be challenging, especially without the guidance of a physiotherapist. They also mentioned that there could be underlying issues, such as an incorrect ACL reconstruction or the growth of soft tissue around the reconstruction, that may be contributing to your limited mobility. Unfortunately, these issues may not be resolvable with exercises and therapy alone.\n\nYour doctor would like to discuss your case further and explore possible options for improving your knee mobility. They will likely need to assess your knee and review your medical history to determine the best course of action.\n\nPlease let us know if you have any questions or concerns, and we will be happy to schedule a follow-up appointment to discuss your treatment options.\n\nBest regards,\n[Your Name]\nOn behalf of [Doctor's Name]", "other_claims": ["Assessing the knee may be necessary to determine the best course of action."], "raw_claim_quality_response": "", "claim_quality_type": "Parse Failed"}

context = contentMap["context"]
sentence = contentMap["sentence"]
# claim = contentMap["claim"]
claim = "Reviewing the medical history is likely necessary to determine the best course of action."
# other_claims = contentMap["other_claims"]
other_claims = ["Assessing the knee is likely necessary to determine the best course of action."]

print('Context:', context)
print('Sentence:', sentence)
print('Claim:', claim)
print('Other Claims:', other_claims)

content = format_input(context=context, sentence=sentence, claim=claim, other_claims=other_claims)
print(content)

result = ollama.chat(
    model='gpt-oss:20b',
    messages=[{"role": "user", "content": content}],
    options={
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 4096,
    }
)

print(result)
with open('test_classification_output.jsonl', 'w') as f:
    f.write(json.dumps({"context": context, "sentence": sentence, "claim": claim, "other_claims": other_claims,
                        "response": result.message.content}) + '\n')
