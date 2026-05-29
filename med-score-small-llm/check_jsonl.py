# load jsonl file and print
import json
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
if __name__ == "__main__":
    file_path = "/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/dmm_small_vision11b/small_llm_provided_decompositions.jsonl"
    data = load_jsonl(file_path)
    print(len(data))
    for item in data[:5]:  # print first 5 items
        print(json.dumps(item, indent=2))







"""Reasoning:
Step 1: TRIAGE (Filter Narratives): The sentence "I spoke to your doctor..." describes a personal interaction. This is an unverifiable narrative. The process stops here.
Facts:
- No verifiable claim

However, since the instruction is to decompose the sentence into atomic facts, we will continue with the rest of the sentence.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "I spoke to your doctor and they wanted to". The core medical content is: "reassure you that your symptoms are not immediately concerning."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "your symptoms" refers to "the swollen lymph node in your neck".
- "you" refers to "the patient".
The specific medical content becomes: "[the doctor] wants to reassure [the patient] that [the swollen lymph node in your neck] are not immediately concerning."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "reassure you that your symptoms are not immediately concerning".
Step 5: Check Incomplete Claims: Preserve the modifier "immediately concerning".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claim is a single concept. No subsets found.
Step 8: Coverage Check: The concept of reassurance regarding symptoms is captured.
Facts:
- The doctor wants to reassure the patient that the swollen lymph node in the neck is not immediately concerning.

However, the original instruction was to decompose the entire sentence into atomic facts. Let's continue with the rest of the sentence.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "They believe that". The core medical content is: "the swollen lymph node in your neck is likely a reactive lymph node, which is a common occurrence after a viral illness like the cough you had."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "your neck" refers to "the neck".
- "your cough" refers to "the cough you had".
The specific medical content becomes: "[the swollen lymph node in your neck] is likely a reactive lymph node, which is a common occurrence after a viral illness like [the cough you had]."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "the swollen lymph node in your neck is likely a reactive lymph node".
Step 5: Check Incomplete Claims: Preserve the modifier "likely".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claim is a single concept. No subsets found.
Step 8: Coverage Check: The concept of the swollen lymph node being a reactive lymph node is captured.
Facts:
- The swollen lymph node in the neck is likely a reactive lymph node.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "The fact that". The core medical content is: "the rest of your neck feels fine and your blood work from a few months ago was normal also supports this diagnosis."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "your neck" refers to "the neck".
- "your blood work" refers to "the blood work from a few months ago".
The specific medical content becomes: "[the rest of the neck feels fine] and [your blood work from a few months ago was normal] also supports this diagnosis."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "the rest of your neck feels fine".
- The main concept is "your blood work from a few months ago was normal".
Step 5: Check Incomplete Claims: Preserve the modifier "also supports".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claims are distinct. No subsets found.
Step 8: Coverage Check: The concepts of neck feeling fine and blood work being normal are captured.
Facts:
- The rest of the neck feels fine.
- Your blood work from a few months ago was normal.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "Your doctor understands that". The core medical content is: "your family history of cancer may be causing you additional worry, but at this point, they do not think it's a red flag in your case."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "your family history of cancer" refers to "the family history of cancer".
- "you" refers to "the patient".
The specific medical content becomes: "[the doctor] understands that [the patient]'s family history of cancer may be causing [the patient] additional worry, but at this point, [the doctor] do not think it's a red flag in [the patient]'s case."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "your family history of cancer may be causing you additional worry".
- The main concept is "they do not think it's a red flag in your case".
Step 5: Check Incomplete Claims: Preserve the modifier "may be causing".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claims are distinct. No subsets found.
Step 8: Coverage Check: The concepts of family history of cancer being a worry and not being a red flag are captured.
Facts:
- The patient's family history of cancer may be causing additional worry.
- The doctor does not think the family history of cancer is a red flag in the patient's case.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "They recommend". The core medical content is: "monitoring the lymph node for any changes in size or if you develop more lymph nodes without an infectious illness."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "the lymph node" refers to "the swollen lymph node in your neck".
- "you" refers to "the patient".
The specific medical content becomes: "[the doctor] recommends monitoring [the swollen lymph node in your neck] for any changes in size or if [the patient] develop more lymph nodes without an infectious illness."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "monitoring the lymph node for any changes in size".
- The main concept is "if you develop more lymph nodes without an infectious illness".
Step 5: Check Incomplete Claims: Preserve the modifier "without an infectious illness".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claims are distinct. No subsets found.
Step 8: Coverage Check: The concepts of monitoring the lymph node and developing more lymph nodes without an infectious illness are captured.
Facts:
- The doctor recommends monitoring the swollen lymph node in the neck for any changes in size.
- The doctor recommends monitoring for developing more lymph nodes without an infectious illness.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "If that happens". The core medical content is: "further investigation may be necessary."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- No references need to be resolved.
The specific medical content becomes: "further investigation may be necessary."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "further investigation may be necessary".
Step 5: Check Incomplete Claims: Preserve the modifier "may be necessary".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claim is a single concept. No subsets found.
Step 8: Coverage Check: The concept of further investigation being necessary is captured.
Facts:
- Further investigation may be necessary.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "They want to reassure you that". The core medical content is: "they are confident in their assessment and would like to continue monitoring your symptoms."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "you" refers to "the patient".
- "they" refers to "the doctor".
The specific medical content becomes: "[the doctor] wants to reassure [the patient] that [the doctor] are confident in their assessment and would like to continue monitoring [the patient]'s symptoms."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "they are confident in their assessment".
- The main concept is "would like to continue monitoring your symptoms".
Step 5: Check Incomplete Claims: Preserve the modifier "would like to continue".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claims are distinct. No subsets found.
Step 8: Coverage Check: The concepts of the doctor being confident in their assessment and continuing to monitor symptoms are captured.
Facts:
- The doctor is confident in their assessment.
- The doctor would like to continue monitoring the patient's symptoms.

Reasoning:
Step 1: TRIAGE (Filter Narratives): It does not include event narratives or present for patient-empathy, and it contains verifiable medical content, so we proceed to Step 2.
Step 2: ISOLATE & FORMAT (Structure Check): I am removing the reporting frame "If you have any further concerns". The core medical content is: "please don't hesitate to reach out to them."
Step 3: DECONTEXTUALIZE (Resolve References): I am resolving references using the Context.
- "you" refers to "the patient".
- "them" refers to "the doctor".
The specific medical content becomes: "[the patient] have any further concerns, please don't hesitate to reach out to [the doctor]."
Step 4: DECOMPOSE (Split Atomic Facts): I am breaking down the content.
- The main concept is "please don't hesitate to reach out to them".
Step 5: Check Incomplete Claims: Preserve the modifier "please don't hesitate".
Step 6: Check Hallucinations: No outside information added. All definitions are strictly from the text.
Step 7: Check Redundant Claims: The claim is a single concept. No subsets found.
Step 8: Coverage Check: The concept of reaching out to the doctor is captured.
Facts:
- Please don't hesitate to reach out to the doctor.

Facts:
- The doctor wants to reassure the patient that the swollen lymph node in the neck is not immediately concerning.
- The swollen lymph node in the neck is likely a reactive lymph node.
- The rest of the neck feels fine.
- Your blood work from a few months ago was normal.
- The patient's family history of cancer may be causing additional worry.
- The doctor does not think the family history of cancer is a red flag in the patient's case.
- The doctor recommends monitoring the swollen lymph node in the neck for any changes in size.
- The doctor recommends monitoring for developing more lymph nodes without an infectious illness.
- Further investigation may be necessary.
- The doctor is confident in their assessment.
- The doctor would like to continue monitoring the patient's symptoms.
- Please don't hesitate to reach out to the doctor."""