import json
from typing import List


def residual_extraction(context: str, sentence: str, current_claims: List[str]) -> str:
    formatted_current_claims = json.dumps(current_claims)
    prompt = f"""You are a specialized Medical Gap Analyst. Your task is to identify medical information missing from a list of "Current Claims" compared to the "Original Sentence", and generate new atomic facts to fill those gaps.

INPUT DATA:
- Context: Full text (used ONLY to resolve pronouns/references).
- Sentence: The ground truth source sentence.
- Current Claims: A list of facts already extracted from the Sentence.

---
INSTRUCTIONS:
1. GAP ANALYSIS: Compare the Sentence vs. Current Claims. Identify specific medical concepts that are present in the Sentence but COMPLETELY ABSENT or INCOMPLETELY REPRESENTED in the Claims.
   - Focus on: Missing Entities (e.g., Drugs, Symptoms), Missing Modifiers (e.g., Severe, Acute), Missing Values (e.g., 500mg, Daily), Missing Conditions (e.g, If, When).
2. IGNORE SYNONYMS: If the meaning is already covered by different words (e.g., "walk" vs "ambulate"), DO NOT create a new claim.
3. GENERATE RECOVERY CLAIMS: Convert the identified missing info into new Atomic Facts.
   - Use the Decomposition Rules by following these guidelines:
        a. Filter Unverifiable Narratives: Ignore event narratives and patient-empathy.
        b. No Hallucinations: Only include information explicitly stated in the Sentence.
        c. Preserve Modifiers: Never drop "may", "could", "likely", dosages, or frequencies; Do not lose the original dependency condition or modifier.
        d. Correct Structure: Convert commands or questions into declarative statements. Remove reporting frames.
        e. Resolve References: Replace all pronouns (he, it, they), all possessive adjectives (their, your, its, his, her) with specific ones from the Context.
        f. No Redundancies: Each fact must be unique and non-overlapping both semantically and wording.
        g. Atomic Concepts: Each fact must contain only ONE medical concept.
        h. No Complex Claims: Split compound sentences containing "AND/OR".
   - If a modifier is missing (e.g., Claim says "Take drug" but Sentence says "Take drug with food"), create a specific fact for the missing part (e.g., "Take drug with food" or "Instruction is to take with food").

---
REASONING CHAIN OF THOUGHT:
Step 1: AUDIT (Compare Sentence vs. Claims)
- Read the Sentence. Check off every piece of information that is already fully covered by the Current Claims.
- What is left over? (The Residuals).

Step 2: IDENTIFY GAPS
- Is there a missing measurement? (e.g., "500mg")
- Is there a missing condition? (e.g., "if fever occurs")
- Is there a missing item from a list? (e.g., "A, B, and C" - but Claims only have A and B).

Step 3: FORMULATE (Create New Facts)
- Turn the missing details into standalone declarative sentences.
- Resolve any pronouns using the Context.

Step 4: VERIFY
- Ensure the new facts are NOT redundant with the Current Claims.
- Ensure the new facts are grounded in the Sentence (No hallucinations).

---
FEW-SHOT EXAMPLES:

Example 1 (Missing Condition & Measurement):
Context: Patient prescribed Amoxicillin.
Sentence: Take Amoxicillin 500mg every 8 hours if symptoms persist.
Current Claims:
- Patient takes Amoxicillin.
- Frequency is every 8 hours.
Reasoning:
Step 1: AUDIT. "Amoxicillin" is covered. "Every 8 hours" is covered.
Step 2: GAPS. "500mg" is missing. "if symptoms persist" is missing.
Step 3: FORMULATE.
  - Gap 1: Dosage is 500mg.
  - Gap 2: Take if symptoms persist.
Facts:
- Amoxicillin dosage is 500mg.
- Take Amoxicillin if symptoms persist.

Example 2 (Missing Item in List):
Context: Reviewing symptoms.
Sentence: Patient reports cough, fever, and shortness of breath.
Current Claims:
- Patient reports cough.
- Patient reports fever.
Reasoning:
Step 1: AUDIT. Cough covered. Fever covered.
Step 2: GAPS. "Shortness of breath" is completely missing.
Step 3: FORMULATE. Patient reports shortness of breath.
Facts:
- Patient reports shortness of breath.

Example 3 (Full Coverage - No Gaps):
Context: Diabetes management.
Sentence: Metformin helps control blood sugar.
Current Claims:
- Metformin helps control blood sugar.
Reasoning:
Step 1: AUDIT. All concepts covered.
Step 2: GAPS. None.
Facts:
- No missing claim

Example 4 (Missing Negation - Critical):
Context: Chest exam.
Sentence: Patient denies chest pain.
Current Claims:
- Patient discusses chest pain.
Reasoning:
Step 1: AUDIT. "Discusses" does not capture the negation "denies". This is a critical gap.
Step 2: GAPS. The negation is missing/incorrect.
Step 3: FORMULATE. Patient denies chest pain.
Facts:
- Patient denies chest pain.

---
YOUR TASK:

Context: {context}

Sentence: {sentence}

Current Claims:
{formatted_current_claims}

Facts (New Recovery Claims Only):"""
    return prompt


import ollama

contentMap = {"id": "ywfe0t_20241101", "sentence_id": 8, "sentence": "They suggest that you discuss the following options with your doctor: requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.", "claim": "Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.", "claim_id": 0, "model_response": "Reasoning:\nStep 1: TRIAGE. The frame \"They suggest that...\" indicates a factual report. I will proceed.\nStep 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is \"They suggest that\". The core medical content is: \"requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.\"\n\nStep 3: DECONTEXTUALIZE. \"these options\" is replaced with its specific meaning from the context: \"the following options\". \"your doctor\" is replaced with \"the doctor\".\nStep 4: DECOMPOSE. I will break the content into all its distinct concepts:\n - Concept 1: Requesting pre-procedural analgesia for anxiety or pain.\n - Concept 2: Asking the doctor to prescribe an opioid for pain management, with a condition.\n - Concept 3: The possibility of having the procedure done under anesthesia in the operating room.\nStep 5: RECONSTRUCT. I will build three declarative sentences, preserving the original context.\n - Fact 1: Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.\n - Fact 2: Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.\n - Fact 3: Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN.\nStep 6: REVIEW - Final Quality and Coverage Check.\n - Deduplication Check: The three facts are distinct and non-redundant.\n - Coverage Check: I've compared the facts to the content from Step 2. All key concepts (the three options for pain management) have been extracted. No important information was omitted.\n\nFacts:\n- Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.\n- Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.\n- Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN.", "context": "I spoke to your doctor and they strongly recommend that you do not decline your Pap smears. According to the Canadian cervical cancer screening guidelines, a Pap test is recommended every three years. However, your doctor notes that the US guidelines suggest that individuals between 30-65 years old can have a Pap test every 5 years with HPV testing, and if the results are negative, the test can be done every 5 years. Your doctor is open to discussing this option with your gynecologist, who can document your agreement to follow the US guidelines and understand the potential risks of fewer Pap tests.\n\nYour doctor emphasizes that while HPV is the primary cause of cervical cancer, it is not the only cause. As you are taking Humira, which suppresses your immune system, you are at a higher risk for cancer. Therefore, it is crucial to detect potential precancerous cells early to prevent them from spreading.\n\nRegarding your concerns about pain during the Pap test, your doctor understands that your history of rectal surgery and fistulas due to Crohn's disease may make the procedure more uncomfortable. They suggest that you discuss the following options with your doctor: requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.\n\nYour doctor encourages you to discuss these options with your healthcare provider to find a solution that works for you.", "other_claims": ["Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.", "Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN."], "raw_claim_quality_response": "Reasoning:\nStep 1: The claim is not a personal narrative or patient‑specific experience; it is a general recommendation.  \nStep 2: The claim is a directive (“Discuss … with your doctor”), which is a command. According to the guidelines, a command is considered incorrectly structured.  \nStep 3–7: Not evaluated because Step 2 already applies.\n\nClassification: Incorrectly structured", "claim_quality_type": "Incorrectly structured"}

context = contentMap["context"]
sentence = contentMap["sentence"]
claim = contentMap["claim"]
error_label = contentMap["claim_quality_type"]

print('Context:', context)
print('Sentence:', sentence)
print('Claim:', claim)
print('Error Label:', error_label)

content = residual_extraction(context=context, sentence=sentence, claim=claim, error_label=error_label)
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
with open('test_normalization_output.jsonl', 'w') as f:
    f.write(json.dumps({"context": context, "sentence": sentence, "claim": claim, "error_label": error_label,
                        "response": result.message.content}) + '\n')
