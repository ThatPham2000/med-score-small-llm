import json


def format_normalization_input(context: str, sentence: str, claim: str, error_label: str) -> str:
    prompt = f"""You are an expert Medical Text Normalization System. Your task is to REPAIR an "Invalid Atomic Claim" based on its "Error Label" and the "Original Source" information.

Your goal is to transform the Invalid Claim into a Valid, Standalone, Declarative, and Complete atomic fact without altering the original medical meaning.

INPUT DATA:
- Context: The full paragraph containing the information.
- Original Sentence: The specific sentence the claim was extracted from.
- Invalid Claim: The claim that needs fixing.
- Error Label: The specific reason the claim was rejected (Context-dependent, Incorrectly structured, or Incomplete).

---
REPAIR STRATEGIES (Follow the strategy corresponding to the Error Label):

STRATEGY A: IF Error Label is "Context-dependent"
- Problem: The claim contains unresolved pronouns (it, they, he, she) or unresolved possessive adjectives (their, your, its, his, her) or vague references (the medication, this condition) or the first person.
- Fix: Locate the specific entity (noun/proper noun) in the "Context" or "Original Sentence" that the pronoun refers to. Replace the pronoun/vague term with the specific entity.
- Example: "It causes nausea" + Context "Metformin..." -> "Metformin causes nausea."

STRATEGY B: IF Error Label is "Incorrectly structured"
- Problem: The claim is an Imperative (command), a Question, or contains a Reporting Frame (e.g., "The doctor said that...").
- Fix (Imperative/Question): Convert the command into a passive advice statement or a declarative fact. Consider the context, use phrases like "Patients should..." or correct ones.
- Fix (Reporting Frame): Remove the reporting frame (e.g., "The study shows that", "The doctor noted") and keep only the core medical fact.
- Example: "Take with food" -> "The medication should be taken with food."
- Example: "The doctor noted that Aspirin reduces pain" -> "Aspirin reduces pain."

STRATEGY C: IF Error Label is "Incomplete"
- Problem: The claim has lost a critical modifier (may, likely), a condition (if, when), or a dependency due to over-decomposition.
- Fix: Retrieve the missing modifier, condition, or clause from the "Original Sentence" and re-attach it to the claim.
- Example: Claim "Stop taking the drug" + Original "If rash occurs, stop taking..." -> "Patients should stop taking the drug if a rash occurs."

---
REASONING PROCESS:
Step 1: Analyze the Error: Identify clearly what makes the claim invalid based on the label.
Step 2: Locate Source Info: Find the missing context, entity, or modifier in the "Context" or "Original Sentence".
Step 3: Draft Repair: Apply the specific Fix Strategy.
Step 4: Final Check: Ensure the new claim is now Valid (Standalone, Declarative, Complete) and does not hallucinate new info.

---
FEW-SHOT EXAMPLES:

Example 1: Repairing "Context-dependent"
Context: "Metformin is the first-line treatment for type 2 diabetes. However, it often causes gastrointestinal issues."
Original Sentence: "However, it often causes gastrointestinal issues."
Invalid Claim: "It often causes gastrointestinal issues."
Error Label: Context-dependent
Reasoning:
Step 1: Analyze the Error: The claim uses "It", which is vague.
Step 2: Locate Source Info: In the context, "It" refers to "Metformin".
Step 3: Draft Repair: Replace "It" with "Metformin".
Step 4: Final Check: "Metformin often causes gastrointestinal issues." is valid.
Normalized Claim: Metformin often causes gastrointestinal issues.

Example 2: Repairing "Context-dependent" (Possessive Adjective + The first person)
Context: "I was prescribed Lisinopril for hypertension. My blood pressure improved significantly."
Original Sentence: "My blood pressure improved significantly."
Invalid Claim: "My blood pressure improved significantly."
Error Label: Context-dependent
Reasoning:
Step 1: Analyze the Error: The claim uses "My", which is first-person possessive.
Step 2: Locate Source Info: "My" refers to the patient's blood pressure.
Step 3: Draft Repair: Replace "My" with "The patient's".
Step 4: Final Check: "The patient's blood pressure improved significantly." is valid.
Normalized Claim: The patient's blood pressure improved significantly.

Example 3: Repairing "Incorrectly structured" (Imperative)
Context: "Managing acid reflux involves lifestyle changes. Avoid eating heavy meals right before bedtime."
Original Sentence: "Avoid eating heavy meals right before bedtime."
Invalid Claim: "Avoid eating heavy meals right before bedtime."
Error Label: Incorrectly structured
Reasoning:
Step 1: Analyze the Error: The claim is an imperative command ("Avoid...").
Step 2: Locate Source Info: The context implies this is advice for managing acid reflux.
Step 3: Draft Repair: Convert to declarative advice. "Patients should avoid eating..."
Step 4: Final Check: The sentence is now a declarative statement.
Normalized Claim: Patients should avoid eating heavy meals right before bedtime.

Example 4: Repairing "Incorrectly structured" (Reporting Frame)
Context: "A recent study verified the effects of the drug. The study shows that Ibuprofen reduces inflammation."
Original Sentence: "The study shows that Ibuprofen reduces inflammation."
Invalid Claim: "The study shows that Ibuprofen reduces inflammation."
Error Label: Incorrectly structured
Reasoning:
Step 1: Analyze the Error: The claim includes the frame "The study shows that".
Step 2: Locate Source Info: Core fact is "Ibuprofen reduces inflammation".
Step 3: Draft Repair: Remove the frame.
Step 4: Final Check: The core fact is preserved.
Normalized Claim: Ibuprofen reduces inflammation.

Example 5: Repairing "Incomplete" (Missing Condition)
Context: "Dosing instructions for Amoxicillin. If you have kidney disease, the dosage requires adjustment."
Original Sentence: "If you have kidney disease, the dosage requires adjustment."
Invalid Claim: "The dosage requires adjustment."
Error Label: Incomplete
Reasoning:
Step 1: Analyze the Error: The claim states dosage adjustment is needed but misses the condition "If you have kidney disease". This makes the advice dangerous/incorrect generally.
Step 2: Locate Source Info: The condition "If you have kidney disease" is in the Original Sentence.
Step 3: Draft Repair: Re-attach the condition.
Step 4: Final Check: The claim now accurately reflects the medical condition.
Normalized Claim: The dosage of Amoxicillin requires adjustment if the patient has kidney disease.

Example 6: Repairing "Incomplete" (Missing Modifier)
Context: "Anabolic steroids may have positive effects on muscle health, but they also carry significant risks."
Original Sentence: "Anabolic steroids may have positive effects on muscle health, but they also carry significant risks."
Invalid Claim: "Anabolic steroids have positive effects on muscle health."
Error Label: Incomplete
Reasoning:
Step 1: Analyze the Error: The claim drops the critical modifier "may", changing the meaning.
Step 2: Locate Source Info: The modifier "may" is present in the Original Sentence.
Step 3: Draft Repair: Re-attach the modifier "may".
Step 4: Final Check: The claim is now complete and accurate.
Normalized Claim: Anabolic steroids may have positive effects on muscle health.

---
OUTPUT FORMAT:
Reasoning:
Step 1: Analyze the Error: [Step 1 reasoning]
Step 2: Locate Source Info: [Step 2 reasoning]
Step 3: Draft Repair: [Step 3 reasoning]
Step 4: Final Check: [Step 4 reasoning]
Normalized Claim: (Your repaired atomic claim)

---
YOUR TASK:

Context: {context}
Original Sentence: {sentence}
Invalid Claim: {claim}
Error Label: {error_label}

Output ONLY the Reasoning and the Final Normalized Claim.
"""
    return prompt


import ollama

# contentMap = {"id": "ywfe0t_20241101", "sentence_id": 8, "sentence": "They suggest that you discuss the following options with your doctor: requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.", "claim": "Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.", "claim_id": 0, "model_response": "Reasoning:\nStep 1: TRIAGE. The frame \"They suggest that...\" indicates a factual report. I will proceed.\nStep 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is \"They suggest that\". The core medical content is: \"requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.\"\n\nStep 3: DECONTEXTUALIZE. \"these options\" is replaced with its specific meaning from the context: \"the following options\". \"your doctor\" is replaced with \"the doctor\".\nStep 4: DECOMPOSE. I will break the content into all its distinct concepts:\n - Concept 1: Requesting pre-procedural analgesia for anxiety or pain.\n - Concept 2: Asking the doctor to prescribe an opioid for pain management, with a condition.\n - Concept 3: The possibility of having the procedure done under anesthesia in the operating room.\nStep 5: RECONSTRUCT. I will build three declarative sentences, preserving the original context.\n - Fact 1: Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.\n - Fact 2: Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.\n - Fact 3: Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN.\nStep 6: REVIEW - Final Quality and Coverage Check.\n - Deduplication Check: The three facts are distinct and non-redundant.\n - Coverage Check: I've compared the facts to the content from Step 2. All key concepts (the three options for pain management) have been extracted. No important information was omitted.\n\nFacts:\n- Discuss requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain with your doctor.\n- Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.\n- Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN.", "context": "I spoke to your doctor and they strongly recommend that you do not decline your Pap smears. According to the Canadian cervical cancer screening guidelines, a Pap test is recommended every three years. However, your doctor notes that the US guidelines suggest that individuals between 30-65 years old can have a Pap test every 5 years with HPV testing, and if the results are negative, the test can be done every 5 years. Your doctor is open to discussing this option with your gynecologist, who can document your agreement to follow the US guidelines and understand the potential risks of fewer Pap tests.\n\nYour doctor emphasizes that while HPV is the primary cause of cervical cancer, it is not the only cause. As you are taking Humira, which suppresses your immune system, you are at a higher risk for cancer. Therefore, it is crucial to detect potential precancerous cells early to prevent them from spreading.\n\nRegarding your concerns about pain during the Pap test, your doctor understands that your history of rectal surgery and fistulas due to Crohn's disease may make the procedure more uncomfortable. They suggest that you discuss the following options with your doctor: requesting pre-procedural analgesia, such as Xanax, to help with anxiety or pain; asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home; if you are seeing an OB/GYN, you can ask about the possibility of having the procedure done under anesthesia in the operating room.\n\nYour doctor encourages you to discuss these options with your healthcare provider to find a solution that works for you.", "other_claims": ["Discuss asking your doctor if they would be willing to prescribe an opioid for pain management, provided you have someone to drive you home.", "Discuss the possibility of having the procedure done under anesthesia in the operating room with your OB/GYN."], "raw_claim_quality_response": "Reasoning:\nStep 1: The claim is not a personal narrative or patient‑specific experience; it is a general recommendation.  \nStep 2: The claim is a directive (“Discuss … with your doctor”), which is a command. According to the guidelines, a command is considered incorrectly structured.  \nStep 3–7: Not evaluated because Step 2 already applies.\n\nClassification: Incorrectly structured", "claim_quality_type": "Incorrectly structured"}
contentMap = {"id": "zcdwtc_20241101", "sentence_id": 1, "sentence": "They believe that the swollen lymph node in your neck is likely a reactive lymph node, which is a common occurrence after a viral illness like the cough you had.", "claim": "A reactive lymph node is a common occurrence after a viral illness like the cough you had.", "claim_id": 1, "model_response": "**Reasoning:**\n\n**Step 1: TRIAGE (Filter Narratives)**\n- The sentence does not include event narratives or patient-empathy. It contains verifiable medical content related to a diagnosis and its reasoning, so we proceed to Step 2.\n\n**Step 2: ISOLATE & FORMAT (Structure Check)**\n- Remove the reporting frame \"They believe that\". The core medical content is: \"the swollen lymph node in your neck is likely a reactive lymph node, which is a common occurrence after a viral illness like the cough you had.\"\n\n**Step 3: DECONTEXTUALIZE (Resolve References)**\n- Resolve references using the Context:\n  - \"your neck\" refers to \"the neck\".\n  - \"the cough you had\" refers to \"a viral illness like the cough you had\".\n- The specific medical content becomes: \"The swollen lymph node in [the neck] is likely a reactive lymph node, which is a common occurrence after [a viral illness like the cough you had].\"\n\n**Step 4: DECOMPOSE (Split Atomic Facts)**\n- Decompose based on conjunctions and definitions:\n  - The main claim about the lymph node being \"likely a reactive lymph node\" stands alone.\n  - The definition of \"reactive lymph node\" as a common occurrence after a viral illness is another fact.\n\n**Step 5: Check Incomplete Claims**\n- Preserve the modifier \"likely\" for the diagnosis.\n- Ensure the full definition of \"reactive lymph node\" is included.\n\n**Step 6: Check Hallucinations**\n- All claims are grounded in the original text. No information was added or inferred beyond what is explicitly stated.\n\n**Step 7: Check Redundant Claims**\n- The facts regarding the diagnosis and its explanation are distinct (Split Exception applies). No subsets found.\n- Ensure no redundancy between the diagnosis and its definition.\n\n**Step 8: Coverage Check**\n- All concepts (diagnosis, likelihood, cause) are captured. Coverage is complete.\n\n**Facts:**\n\n- The swollen lymph node in the neck is likely a reactive lymph node.\n- A reactive lymph node is a common occurrence after a viral illness like the cough you had.", "context": "I spoke to your doctor and they wanted to reassure you that your symptoms are not immediately concerning. They believe that the swollen lymph node in your neck is likely a reactive lymph node, which is a common occurrence after a viral illness like the cough you had. The fact that the rest of your neck feels fine and your blood work from a few months ago was normal also supports this diagnosis.\n\nYour doctor understands that your family history of cancer may be causing you additional worry, but at this point, they do not think it's a red flag in your case. They recommend monitoring the lymph node for any changes in size or if you develop more lymph nodes without an infectious illness. If that happens, further investigation may be necessary.\n\nFor now, your doctor thinks that the risks of further imaging and potential biopsy outweigh the benefits. They want to reassure you that they are confident in their assessment and would like to continue monitoring your symptoms. If you have any further concerns or notice any changes, please don't hesitate to reach out to them.", "other_claims": ["The swollen lymph node in the neck is likely a reactive lymph node."], "raw_claim_quality_response": "Reasoning:\nStep 1: The claim is a medical statement, not a personal narrative or empathetic remark.  \nStep 2: It is a declarative sentence with no reporting frame, question, or command.  \nStep 3: The claim contains the phrase \u201cthe cough you had,\u201d which refers to a specific event outside the claim itself. A reader would need context to know which cough is meant. Therefore the claim is context\u2011dependent.  \nStep 4: Not reached because the claim was classified as context\u2011dependent.  \n\nClassification: Context-dependent", "claim_quality_type": "Context-dependent"}

context = contentMap["context"]
sentence = contentMap["sentence"]
claim = contentMap["claim"]
error_label = contentMap["claim_quality_type"]

print('Context:', context)
print('Sentence:', sentence)
print('Claim:', claim)
print('Error Label:', error_label)

content = format_normalization_input(context=context, sentence=sentence, claim=claim, error_label=error_label)
print(content)

result = ollama.chat(
    model='ministral-3:14b',
    messages=[{"role": "user", "content": content}],
    options={
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 10240,
    }
)

print(result)
with open('test_normalization_output.jsonl', 'w') as f:
    f.write(json.dumps({"context": context, "sentence": sentence, "claim": claim, "error_label": error_label,
                        "response": result.message.content}) + '\n')
