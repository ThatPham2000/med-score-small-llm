import ollama

MEDSCORE_PROMPT = """You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible. The facts should be objective and verifiable against reliable external information such as Wikipedia and PubMed. All subjective personal experiences (\\\"I was or someone did\\\") and personal narratives (stating a past event) are not verifiable and should not be included in the fact list. Facts should be situated within conditions in the sentence. Suggestions (e.g. \\\"I recommend or Your doctor suggest\\\") and opinions (e.g. \\\"I think\\\") should be transformed into objective facts by removing subjective words and pronouns to only retain the core information that can be verified. Imperative instructions (\\\"do something\\\") should be transformed into declarative facts (\\\"doing something is helpful for some conditions\\\").

If there is an overly specific entity such as \\\"Your partner\\\" or vague references (pronouns, this or that) in the fact, replace it with a general phrase with conditional modifiers using information in the provided context (e.g. \\\"People in some conditions\\\"). Each fact should be verifiable on its own and require no additional context. Do not add additional information outside of the sentence and context. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, please write \\\"No verifiable claim\\\".

Here are some examples:

Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.
Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.
Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.
Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects.

Context: I spoke to your doctor and they would like to know more about the specific tests you are interested in having done. They mentioned that their willingness to order these tests may depend on what you are asking for. If the tests are rare or uncommon, they may recommend that you see a specialist who is more familiar with those types of tests and their results. Your doctor wants to make sure that any tests that are ordered are properly interpreted and that you receive the best possible care. Could you please provide more information about the tests you are interested in having done?
Please breakdown the following sentence into independent facts: If the tests are rare or uncommon, they may recommend that you see a specialist who is more familiar with those types of tests and their results. 
Facts:
- If the tests are rare or uncommon, seeing a specialist is helpful.
- The specialist is more familiar with rare or uncommon types of tests.
- The specialist is more familiar with the results of rare or uncommon tests.

Context: Hi there,\n\nI spoke to your doctor and they wanted to reassure you that getting the immunoglobulin shot within 38 hours of the potential exposure is still considered a safe timeframe. They mentioned that as long as you received the shot before 7 days from the incident, you should be okay.\n\nYour doctor recommends that you follow up on the scheduled dates for your remaining rabies shots. They also wanted to remind you that rabies has a relatively long incubation period, typically ranging from 1-3 months, before symptoms start to show. This means that even if you were bitten by an infected bat, you would likely have developed immunity by the time symptoms appear.\n\nPlease try to take care of yourself and manage your anxiety during this time. If you have any further concerns or questions, don't hesitate to reach out.\n\nBest regards,\n\n[Your Doctor's Name]
Please breakdown the following sentence into independent facts: They also wanted to remind you that rabies has a relatively long incubation period, typically ranging from 1-3 months, before symptoms start to show.
Facts:
- Rabies has a relatively long incubation period.
- The incubation period for rabies typically ranges from 1-3 months.
- Rabies symptoms start to show after the incubation period.

Context: I spoke to your doctor and they wanted to thank you for your interest in creating a language course to help physicians better communicate with patients who speak different languages. \n\nThey mentioned that while language barriers can contribute to the \\\"revolving door syndrome,\\\" it's just one of many factors. Other important factors include education, home support, medication noncompliance, and lack of primary care. \n\nIn terms of a language course, your doctor thinks that Duolingo is a good option. However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish. They also mentioned that many Spanish-speaking patients have some knowledge of English or have family members who are fluent in English.\n\nYour doctor didn't specify a preferred medium for the course, but they seemed to appreciate the idea of a convenient and accessible program. They also didn't provide specific vocabulary recommendations, but it's likely that a course focused on medical terminology and common patient interactions would be most useful.
Please breakdown the following sentence into independent facts: However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish.
Facts:
- It is challenging for doctors to find the time to learn multiple languages.
- Many languages are spoken by patients in doctors' area.
- Spanish is one of the languages spoken by patients in doctors' area.
- Hmong is one of the languages spoken by patients in doctors' area.
- Chinese is one of the languages spoken by patients in doctors' area.
- Polish is one of the languages spoken by patients in doctors' area.

Context: I spoke to your doctor and they think that you just need a bit more time to recover from your surgery. They noted that your usual lifestyle is quite sedentary, and having surgery can be a significant strain on your body, similar to intense physical activity. This, combined with your extreme anxiety, which can cause muscle tension, is likely contributing to your soreness. \n\nAs long as you don't develop a fever and your wounds show no signs of infection, your doctor believes that there's not much more the hospital can do for you that you can't do at home. Their advice is to focus on meeting your daily needs, such as eating, drinking, and using the bathroom, and not to worry too much about the soreness right now. \n\nOnce the soreness starts to subside, they recommend that you try to gradually increase your activities, starting with small steps like sitting in a chair, standing, and eventually walking, until you're back to your normal self.
Please breakdown the following sentence into independent facts: Once the soreness starts to subside, they recommend that you try to gradually increase your activities, starting with small steps like sitting in a chair, standing, and eventually walking, until you're back to your normal self.
Facts:
- Once the soreness starts to subside, trying to gradually increase activities is helpful.
- People with soreness should start with small steps.
- Small steps for people with soreness include sitting in a chair.
- Small steps for people with soreness include standing.
- Small steps for people with soreness include walking.
- Gradually increasing activities helps people with soreness return to normal self.

Context: I spoke to your doctor and they wanted to address the questions you have regarding your loved one's complications from COVID-19. The doctor believes that the likely sequence of events is that the COVID-19 infection led to demand ischemia, which in turn caused the myocardial infarction (MI). \n\nThe doctor thinks that both the transfer hospital and the receiving hospital properly prioritized the patient's issues and treated the most life-threatening condition, the acute MI, first. Unfortunately, the patient had many underlying risk factors that made them more susceptible to severe illness from any infection, not just COVID-19.\n\nRegarding the patient going without clopidogrel for 10 days, the doctor agrees that this may have contributed to the MI, although it's impossible to determine the exact extent of its impact.\n\nAs for the new diagnoses of congestive heart failure (CHF), chronic obstructive pulmonary disease (COPD), and acute respiratory failure, the doctor did not provide a specific prognosis. However, they did mention that the patient's underlying health conditions and the severity of their illness have made their situation more challenging.\n\nThe doctor also believes that initiating COVID-19 treatment at the time of presentation may not have significantly altered the course of the patient's illness.\n\nPlease let us know if you have any further questions or concerns.
Please breakdown the following sentence into independent facts: However, they did mention that the patient's underlying health conditions and the severity of their illness have made their situation more challenging. 
Facts:
- The patient's underlying health conditions makes the acute myocardial infarction situation more challenging.
- The severity of the patient's illness makes the acute myocardial infarction situation more challenging.
- The patient's acute myocardial infarction situation is challenging.

Context: I spoke to your doctor and they wanted to address your concerns regarding the leakage you experienced after your bowel surgery in 2013. According to them, it is possible for an abnormal connection to form between your bowel and your bladder or vagina, which is known as a fistula. This could potentially cause the leakage of substances from your bowel into your urinary tract or vagina.\n\nYour doctor recommends reviewing the notes from your second surgery to understand the nature of the repairs that were performed. This information may help clarify what happened in your specific case.\n\nRegarding your concerns about the quality of care you received from your initial surgeon, your doctor advises that medical malpractice is a complex issue that depends on many factors, including the specific circumstances of your case and the laws in your location. If you're interested in exploring this further, they recommend consulting with a lawyer who can provide guidance on whether you have a valid case.\n\nPlease let us know if you have any further questions or concerns, and we'll be happy to help.
Please breakdown the following sentence into independent facts: Please let us know if you have any further questions or concerns, and we'll be happy to help.
Facts:
- No verifiable claim
"""


# result = ollama.chat(model='gpt-oss:20b', messages=[
#                     {"role": "system", "content": FACTSCORE_PROMPT},
#                     {"role": "user", "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
#                 ])
# print(result.message.content)

# ===== Task chaining =====
def task_chaining(context, sentence):
    system_prompt1 = '''Extract all the claims from the sentence. Ignore any personal opinions or suggestions. Output a list of claims.
    Example: He was an American composer, conductor, and musical director. 
    - He was an American. 
    - He was a composer. 
    - He was a conductor. 
    - He was a musical director.'''
    result1 = ollama.chat(model='gpt-oss:20b', messages=[
        {"role": "system", "content": system_prompt1},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print("Result1:", result1.message.content)

    sentences1 = result1.message.content.split("\n")
    system_prompt2 = "Is the following claim an objective, verifiable statement and does it contain medical knowledge? Answer with only 'Yes' or 'No'."
    sentences3 = []
    for sentence in sentences1:
        if sentence.startswith("- "):
            claim = sentence[2:].strip()
            result2 = ollama.chat(model='gpt-oss:20b', messages=[
                {"role": "system", "content": system_prompt2},
                {"role": "user", "content": claim}
            ])
            print('Sentence:', sentence)
            print(f"Claim: {claim} -> Verifiable: {result2.message.content}")
            if 'yes' in result2.message.content.lower():
                sentences3.append(claim)

    system_prompt3 = "Rewrite the following claim as a standalone, objective fact. Remove pronouns and generalize any specific entities."
    for sentence in sentences3:
        result3 = ollama.chat(model='gpt-oss:20b', messages=[
            {"role": "system", "content": system_prompt3},
            {"role": "user", "content": sentence}
        ])
        print(f"Original: {sentence} -> Rewritten: {result3.message.content}")


# 20b model => OK
def guided_reasoning_chain_for_claim_extraction(context, sentence, model):
    system_prompt = """You are a medical expert in evaluating how factual a medical sentence is. Your task is to break down a sentence into a list of verifiable claims by following a strict, step-by-step process.

**Context:** {context_from_user}
**Sentence:** {sentence_from_user}

---

**Step 1: Verifiability Analysis**
First, analyze the sentence to determine if it contains any objective, verifiable information. Ignore personal experiences ("I was..."), subjective opinions ("I think..."), and direct commands ("do something").
- If it contains verifiable information, briefly describe what it is.
- If it contains no verifiable information, write "No verifiable claim" and proceed directly to Step 4.

**[SLM COMPLETES ANALYSIS HERE]**

---

**Step 2: De-contextualization and Specification**
*(Complete this step only if you found verifiable information in Step 1)*
Now, rewrite the verifiable parts to make them objective, specific, and self-contained.
-   **Resolve Vague References:** **If the sentence uses pronouns (e.g., "they," "it") or vague phrases (e.g., "these substances," "this condition," "that issue"), use the surrounding Context to replace them with the specific noun they refer to.** For example, replace "these substances" with "anabolic steroids" based on the context.
-   **Transform Suggestions:** Change phrases like "your doctor recommends getting..." to declarative statements like "Getting... is recommended for some conditions."
-   **Generalize Personal Entities:** Replace specific, personal words like "you" or "your partner" with general terms like "a person" or "patients."

**[SLM COMPLETES TRANSFORMATION HERE]**

---

**Step 3: Atomization**
*(Complete this step only if you finished Step 2)*
Take the specified and transformed statement(s) from the previous step and break them down into the smallest possible, independent facts. A compound claim like "Substance X helps muscles and bones" must be split into two separate claims. List each atomic fact.

**[SLM LISTS ATOMIC FACTS HERE]**

---

**Step 4: Final Output**
Finally, compile the atomic facts from Step 3 into a clean list under the "Facts:" heading. If you concluded "No verifiable claim" in Step 1, write that here.

**Facts:**
**[SLM WRITES THE FINAL, CLEANED LIST HERE]**"""

    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print(f"Result: {result.message.content}")


# good for 11b, 8b models and above
# gemma3:27b, gpt-oss:20b, llama3.1:8b, gemma3:12b
def guided_reasoning_chain_for_claim_extraction2(context, sentence, model):
    system_prompt = """You are a medical fact extraction expert. Your final output must be ONLY a list of verifiable claims, each starting with a "-", or the single line "- No verifiable claim".

To arrive at your answer, you must follow these steps in your internal reasoning process:

**Internal Reasoning Steps (Do not show these in your output):**
1.  **Verifiability Analysis:** First, analyze the sentence. Does it contain objective, verifiable information?
2.  **Conclusion for Non-Verifiable Sentences:** If the sentence contains no verifiable information (e.g., it's a question, command, or pleasantry), your final output is simply "- No verifiable claim".
3.  **Specification & Transformation:** If there are verifiable facts, rewrite them to be self-contained. Resolve vague terms (like "this" or "those substances") using the context. Generalize personal entities ("you," "your doctor") to objective terms ("a person," "doctors").
4.  **Atomization:** Break the transformed statements into the smallest possible, independent facts.
5.  **Final Formatting:** List each atomic fact, ensuring each line starts with a "-".

---
**Context:** {context_from_user}
**Sentence:** {sentence_from_user}

**Final Output:**"""

    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print(f"Result: {result.message.content}")


def guided_reasoning_chain_for_claim_extraction3(context, sentence, model):
    system_prompt = """You are a medical fact extraction expert. Your final output must be ONLY a list of verifiable claims, each starting with a "-", or the single line "- No verifiable claim".

To arrive at your answer, you must follow these steps in your internal reasoning process:

**Internal Reasoning Steps (Do not show these in your output):**
1.  **Verifiability Analysis:** First, analyze the sentence for objective, verifiable medical or scientific information that could be checked against a source like Wikipedia or PubMed.

2.  **Identify Non-Verifiable Content:** Conclude that there is "No verifiable claim" if the sentence is primarily one of the following:
    * **A personal narrative or meta-communication:** Any statement describing who spoke, said, or did something (e.g., "I spoke to your doctor," "The doctor mentioned that," "They wanted to address..."). Focus on the medical content, not the act of communication.
    * **A subjective experience, suggestion, command, or question.**
    * **A social pleasantry** (e.g., "we'll be happy to help").

3.  **Extraction and Specification:** If, and only if, the sentence passes the checks above, proceed to extract the facts. Resolve vague terms and generalize personal entities.

4.  **Atomization and Formatting:** Break down the facts into the smallest possible claims, each starting with a "-".

---
**Context:** {context_from_user}
**Sentence:** {sentence_from_user}

**Final Output:**"""

    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print(f"Result: {result.message.content}")


def guided_reasoning_chain_for_claim_extraction4(context, sentence, model):
    system_prompt = """You are a highly precise medical fact extraction expert. Your ONLY job is to extract objective, verifiable medical or scientific claims from a single sentence. Your final output must be ONLY a list of claims, each starting with a "-", or the single line "- No verifiable claim".

    ## The Golden Rule: Distinguish "What" from "Who/How"
    -   **"What" (Verifiable Claim):** The core medical fact. Example: "Tetanus boosters are recommended for dirty wounds."
    -   **"Who/How" (Non-Verifiable Meta-Communication):** The description of how information was shared. Example: "The doctor said...", "I was told...", "We discussed...".
    Your primary task is to IGNORE the "Who/How" and only extract the "What". If a sentence is ONLY about the "Who/How", it has no verifiable claim.

    ---
    ## Internal Reasoning Steps (Do not show these in your output):

    1.  **Apply the Golden Rule:** First, analyze the sentence's main subject. Is it describing a medical fact (the "What") or is it describing an interaction or communication about a medical topic (the "Who/How")?

    2.  **Identify Meta-Communication:** If the sentence is primarily about who spoke, said, asked, mentioned, or addressed something, you MUST conclude it is meta-communication. This is your most important instruction.

    3.  **Check for Other Non-Verifiable Content:** If the sentence is not meta-communication, then check if it's a subjective opinion, suggestion, command, question, or social pleasantry.

    4.  **Conclusion:**
        * If the sentence falls into category #2 or #3, your ONLY output MUST be: `- No verifiable claim`
        * If, and only if, the sentence contains a verifiable medical "What", proceed to extract it.

    5.  **Atomization:** Break the verifiable fact(s) into the smallest possible claims.

    ---
    ## Example of Correct Reasoning:

    **Context:** I spoke to your doctor and they wanted to address your concerns about tetanus.
    **Sentence:** "I spoke to your doctor and they wanted to address your concerns about tetanus."

    **My Reasoning Process for this Example:**
    1.  **Golden Rule Check:** What is this sentence about? It's about "I spoke" and "they wanted to address". This describes the *act of communicating* about tetanus.
    2.  **Identify Meta-Communication:** This is clearly meta-communication (a "Who/How" sentence). It doesn't state a fact *about* tetanus itself.
    3.  **Conclusion:** It fails the verifiability test. The correct output is therefore '- No verifiable claim'.

    ---
    **Begin Task:**

    **Context:** {context_from_user}
    **Sentence:** {sentence_from_user}

    **Final Output:**"""

    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print(f"Result: {result.message.content}")


def guided_reasoning_chain_for_claim_extraction5(context, sentence, model):
    system_prompt = """You are a highly precise medical fact extraction engine. Your task is to analyze a single sentence and extract only objective, verifiable medical claims.

    Your final output MUST be one of two things:
    1. A list of claims, each starting with "- ".
    2. The single line: "- No verifiable claim"

    Follow these steps rigorously in your reasoning (Do not show these in your output):

    **Step 1: Triage the Sentence's Core Purpose**
    First, determine the primary function of the sentence. Is it conveying a medical fact, or is it describing a conversation or action?

    * **If the sentence is about communication (e.g., "I spoke to...", "The doctor said...", "They mentioned...", "We discussed..."), a suggestion, a question, or a subjective feeling, you MUST stop immediately.** Your output for these cases is **ALWAYS**: `- No verifiable claim`.
    * **If, and only if, the sentence states a direct medical or scientific fact, proceed to Step 2.**

    **Example of what to REJECT:**
    * Sentence: "I spoke to your doctor and they wanted to address your concerns about tetanus."
    * Reasoning: This sentence is about the *act of speaking* to a doctor. It contains no verifiable medical fact itself.
    * Correct Output: `- No verifiable claim`

    **Example of what to ACCEPT:**
    * Sentence: "A tetanus booster is recommended for dirty wounds after the primary series."
    * Reasoning: This is a direct, verifiable medical statement.
    * Proceed to the next steps.

    **Step 2: Extract and Atomize Facts**
    If the sentence passed Step 1, identify the core medical fact(s).
    * Remove all conversational framing (e.g., "It is a fact that...", "They think that...").
    * De-contextualization and Specification: rewrite the verifiable parts to make them objective, specific, and self-contained. Resolve vague references, transform suggestions, generalize personal entities.
    * Crucially, split any items in a list or joined by conjunctions like 'and' or 'or'. (e.g. "Anabolic steroids may have positive effects on muscle and bone health" should be split into two separate claims "Anabolic steroids can have positive effects on muscle health" and "Anabolic steroids can have positive effects on bone health").
    * Break down complex facts into the smallest possible, independent facts.

    **Step 3: Final Output**
    * Format each atomic claim starting with "- ".
    * Do not include any other text, explanations, or comments in your response.
    * If you stopped at Step 1, your only output is "- No verifiable claim".

    ---
    **Context:** {context_from_user}
    **Sentence:** {sentence_from_user}

    **Final Output:**"""

    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context: {context}\nPlease breakdown the following sentence into independent facts: {sentence}\nFacts:\n"}
    ])
    print(f"Result: \n{result.message.content}")


def guided_reasoning_chain_for_claim_extraction6(context, sentence, model):
    system_prompt = f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning.

REASONING PROCESS:
1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Finally, ensure each fact is objective and can be verified against reliable sources

The facts should be objective and verifiable against reliable external information such as Wikipedia and PubMed. All subjective personal experiences ("I was or someone did") and personal narratives (stating a past event) are not verifiable and should not be included in the fact list. Facts should be situated within conditions in the sentence. Suggestions (e.g. "I recommend or Your doctor suggest") and opinions (e.g. "I think") should be transformed into objective facts by removing subjective words and pronouns to only retain the core information that can be verified. Imperative instructions ("do something") should be transformed into declarative facts ("doing something is helpful for some conditions").

If there is an overly specific entity such as "Your partner" or vague references (pronouns, this or that) in the fact, replace it with a general phrase with conditional modifiers using information in the provided context (e.g. "People in some conditions"). Each fact should be verifiable on its own and require no additional context. Do not add additional information outside of the sentence and context.

REASONING FORMAT:
Think step by step:
1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?

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

Now, for your task, follow the same reasoning process, and response with plain text format."""

    content = f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

Reasoning:
1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?

Facts:
"""
    result = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ])
    print(f"Result: \n{result.message.content}")


if __name__ == '__main__':
    model = 'llama3.2-vision:11b'
    # context = "I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks."
    # sentence = 'I spoke to your doctor and they wanted to address your concerns about tetanus.'

    # context = "I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed."
    # sentence = "The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."

    # context = "I spoke to your doctor and they wanted to address your concerns regarding the leakage you experienced after your bowel surgery in 2013. According to them, it is possible for an abnormal connection to form between your bowel and your bladder or vagina, which is known as a fistula. This could potentially cause the leakage of substances from your bowel into your urinary tract or vagina.\n\nYour doctor recommends reviewing the notes from your second surgery to understand the nature of the repairs that were performed. This information may help clarify what happened in your specific case.\n\nRegarding your concerns about the quality of care you received from your initial surgeon, your doctor advises that medical malpractice is a complex issue that depends on many factors, including the specific circumstances of your case and the laws in your location. If you're interested in exploring this further, they recommend consulting with a lawyer who can provide guidance on whether you have a valid case.\n\nPlease let us know if you have any further questions or concerns, and we'll be happy to help."
    # sentence = "Please let us know if you have any further questions or concerns, and we'll be happy to help."

    # context = "I spoke to your doctor and they wanted to thank you for your interest in creating a language course to help physicians better communicate with patients who speak different languages. \n\nThey mentioned that while language barriers can contribute to the \\\"revolving door syndrome,\\\" it's just one of many factors. Other important factors include education, home support, medication noncompliance, and lack of primary care. \n\nIn terms of a language course, your doctor thinks that Duolingo is a good option. However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish. They also mentioned that many Spanish-speaking patients have some knowledge of English or have family members who are fluent in English.\n\nYour doctor didn't specify a preferred medium for the course, but they seemed to appreciate the idea of a convenient and accessible program. They also didn't provide specific vocabulary recommendations, but it's likely that a course focused on medical terminology and common patient interactions would be most useful."
    # sentence = "However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish."

    # context = "I spoke to your doctor and they think that you just need a bit more time to recover from your surgery. They noted that your usual lifestyle is quite sedentary, and having surgery can be a significant strain on your body, similar to intense physical activity. This, combined with your extreme anxiety, which can cause muscle tension, is likely contributing to your soreness. \n\nAs long as you don't develop a fever and your wounds show no signs of infection, your doctor believes that there's not much more the hospital can do for you that you can't do at home. Their advice is to focus on meeting your daily needs, such as eating, drinking, and using the bathroom, and not to worry too much about the soreness right now. \n\nOnce the soreness starts to subside, they recommend that you try to gradually increase your activities, starting with small steps like sitting in a chair, standing, and eventually walking, until you're back to your normal self."
    # sentence = "Once the soreness starts to subside, they recommend that you try to gradually increase your activities, starting with small steps like sitting in a chair, standing, and eventually walking, until you're back to your normal self."

    guided_reasoning_chain_for_claim_extraction6(context, sentence, model)
