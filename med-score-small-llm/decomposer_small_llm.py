from typing import List, Dict, Any
import re
from collections import Counter

from decomposer import Decomposer
from llm import LLM


class DecomposerSmallLLM(Decomposer):
    """
    Enhanced decomposer for small language models with improved reasoning capabilities.
    This implementation addresses the 7 MedScoreTaxonomy issues:
    1. Unverifiable claims - filters out personal narratives and patient-specific interactions
    2. Hallucinated claims - validates claims are grounded in original sentence
    3. Incomplete claims - preserves important modifiers and dependencies
    4. Incorrectly structured claims - transforms imperatives to declaratives
    5. Context-dependent claims - handles vague references and pronouns
    6. Redundant claims - implements deduplication mechanism
    7. Omitted claims - ensures comprehensive coverage validation
    """

    def __init__(
            self,
            llm: LLM = None,
            reasoning_steps: int = 3,
            enable_validation: bool = True,
            similarity_threshold: float = 0.8,
    ):
        super().__init__(llm=llm)
        self.reasoning_steps = reasoning_steps
        self.enable_validation = enable_validation
        self.similarity_threshold = similarity_threshold

    def get_system_prompt(self) -> str:
        """Enhanced system prompt addressing the 7 MedScoreTaxonomy issues"""
        # Generate dynamic reasoning steps based on the reasoning_steps parameter
        reasoning_steps_text = self._generate_reasoning_steps()
        reasoning_format = self._generate_reasoning_format()

        # Generate dynamic examples based on reasoning steps
        examples = self._generate_examples()

        return f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning while addressing the 7 MedScoreTaxonomy issues.

REASONING PROCESS:
{reasoning_steps_text}

QUALITY REQUIREMENTS (Addressing 7 MedScoreTaxonomy Issues):

1. UNVERIFIABLE CLAIMS: Filter out personal narratives, patient-specific interactions, and bedside manner statements. Exclude:
   - Personal experiences ("I spoke with your doctor", "you are experiencing pain")
   - Patient-specific interactions
   - Bedside manner statements ("Your pain can be very tiring")

2. HALLUCINATED CLAIMS: Ensure all claims are grounded in the original sentence:
   - No additional information beyond the sentence
   - No distortion of original meaning
   - No irrelevant information

3. INCOMPLETE CLAIMS: Preserve important modifiers and dependencies:
   - Maintain conditional statements and modifiers
   - Keep temporal and contextual information
   - Preserve cause-effect relationships

4. INCORRECTLY STRUCTURED CLAIMS: Transform to declarative format:
   - Convert imperatives ("Take ibuprofen") to declaratives ("Taking ibuprofen is helpful for pain")
   - Remove nested sub-clauses ("They said [claim]" → "[claim]")
   - Ensure declarative sentence structure

5. CONTEXT-DEPENDENT CLAIMS: Handle vague references:
   - Replace pronouns with specific entities from context
   - Clarify temporal references ("this morning" → specific date/time)
   - Specify locations and entities using context information

6. REDUNDANT CLAIMS: Avoid minimal modifications:
   - Don't create multiple versions of the same fact
   - Eliminate repetitive claims
   - Focus on distinct, non-overlapping facts

7. OMITTED CLAIMS: Ensure comprehensive coverage:
   - Extract all important medical information
   - Don't miss key facts from the sentence
   - Maintain completeness of medical content

REASONING FORMAT:
Think step by step:
{reasoning_format}

If there is no verifiable fact in the sentence, please write "No verifiable claim".

Here are some examples with reasoning:

{examples}

Now, for your task, follow the same reasoning process."""

    def format_input(self, context: str, sentence: str) -> str:
        """Enhanced input formatting with reasoning prompts"""
        reasoning_format = self._generate_reasoning_format()
        return f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

Reasoning:
{reasoning_format}

Facts:
"""

    def format_completions(self, decomp_input: List[Dict[str, Any]], completions: List[str]) -> List[Dict[str, Any]]:
        """Enhanced completion formatting that handles reasoning steps and validates claims"""
        decompositions = []
        for d_input, completion in zip(decomp_input, completions):
            # Extract facts from completion, handling reasoning format
            # Find the "Facts:" section and extract claims from there
            lines = completion.split("\n")
            facts_started = False
            claim_list = []

            for line in lines:
                if "Facts:" in line:
                    facts_started = True
                    continue
                if facts_started and line.strip():
                    # Skip reasoning lines that start with numbers or "Reasoning:"
                    # Handle dynamic number of reasoning steps
                    reasoning_patterns = tuple(f"{i}." for i in range(1, self.reasoning_steps + 1))
                    if line.strip().startswith(reasoning_patterns + ("Reasoning:", "Think")):
                        continue
                    # Extract claim from bullet point
                    if line.strip().startswith("- "):
                        claim = line.strip()[2:].strip()
                        if claim and "no verifiable claim" not in claim.lower():
                            claim_list.append(claim)
                    elif line.strip() and not line.strip().startswith(("Context:", "Please breakdown")):
                        # Handle cases where facts don't start with bullet points
                        claim = line.strip()
                        if claim and "no verifiable claim" not in claim.lower():
                            claim_list.append(claim)

            # Apply validation and filtering if enabled
            if self.enable_validation:
                claim_list = self._validate_and_filter_claims(claim_list, d_input.get("sentence", ""), d_input.get("context", ""))

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

    def _generate_examples(self) -> str:
        """Generate dynamic examples based on reasoning steps"""
        if self.reasoning_steps == 1:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What verifiable medical facts can be extracted from this sentence? - None, this is a personal narrative about speaking to a doctor (UNVERIFIABLE)

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What verifiable medical facts can be extracted from this sentence? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects (COMPLETE with modifiers)

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 2:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 3:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 4:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience
4. Are these facts complete and contextually appropriate? - N/A, no verifiable facts

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects
4. Are these facts complete and contextually appropriate? - Yes, they are complete and contextually appropriate

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        elif self.reasoning_steps == 5:
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience
4. Are these facts complete and contextually appropriate? - N/A, no verifiable facts
5. Are these facts accurate and comprehensive? - N/A, no verifiable facts

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects
4. Are these facts complete and contextually appropriate? - Yes, they are complete and contextually appropriate
5. Are these facts accurate and comprehensive? - Yes, they accurately represent the medical information

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

        else:
            # Default to 3 steps for any other value
            return """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. What are the main medical concepts here? - doctor consultation, tetanus concerns
2. What facts can be extracted from each concept? - None, this is a personal narrative about speaking to a doctor
3. Are these facts verifiable and objective? - No, this is a subjective personal experience

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. What are the main medical concepts here? - substances, muscle health, bone health, risks, side effects
2. What facts can be extracted from each concept? - Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. Are these facts verifiable and objective? - Yes, these are objective medical facts about substance effects

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may also carry significant risks.
- Anabolic steroids may carry potential side effects."""

    def _generate_reasoning_steps(self) -> str:
        """Generate dynamic reasoning steps based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. Identify and extract all verifiable medical facts from the sentence"
        elif self.reasoning_steps == 2:
            return """1. First, identify the main medical concepts in the sentence
2. Then, extract verifiable facts from each concept"""
        elif self.reasoning_steps == 3:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Finally, ensure each fact is objective and can be verified against reliable sources"""
        elif self.reasoning_steps == 4:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Next, ensure each fact is objective and can be verified against reliable sources
4. Finally, validate that each fact is complete and contextually appropriate"""
        elif self.reasoning_steps == 5:
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Next, ensure each fact is objective and can be verified against reliable sources
4. Then, validate that each fact is complete and contextually appropriate
5. Finally, review and refine the extracted facts for accuracy and completeness"""
        else:
            # Default to 3 steps for any other value
            return """1. First, identify the main medical concepts in the sentence
2. Then, break down each concept into verifiable facts
3. Finally, ensure each fact is objective and can be verified against reliable sources"""

    def _generate_reasoning_format(self) -> str:
        """Generate dynamic reasoning format based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. What verifiable medical facts can be extracted from this sentence?"
        elif self.reasoning_steps == 2:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?"""
        elif self.reasoning_steps == 3:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?"""
        elif self.reasoning_steps == 4:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?
4. Are these facts complete and contextually appropriate?"""
        elif self.reasoning_steps == 5:
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?
4. Are these facts complete and contextually appropriate?
5. Are these facts accurate and comprehensive?"""
        else:
            # Default to 3 steps for any other value
            return """1. What are the main medical concepts here?
2. What facts can be extracted from each concept?
3. Are these facts verifiable and objective?"""

    def _validate_and_filter_claims(self, claims: List[str], sentence: str, context: str) -> List[str]:
        """Validate and filter claims to address the 7 MedScoreTaxonomy issues"""
        if not claims:
            return claims
            
        validated_claims = []
        
        for claim in claims:
            # 1. Check for unverifiable claims (personal narratives, patient interactions)
            if self._is_unverifiable_claim(claim):
                continue
                
            # 2. Check for hallucinated claims (not grounded in sentence)
            if not self._is_grounded_in_sentence(claim, sentence):
                continue
                
            # 3. Check for incomplete claims (missing important modifiers)
            if self._is_incomplete_claim(claim, sentence):
                continue
                
            # 4. Transform incorrectly structured claims
            claim = self._transform_to_declarative(claim)
            
            # 5. Handle context-dependent claims (vague references)
            claim = self._resolve_context_dependencies(claim, context)
            
            # 6. Check for redundant claims
            if self._is_redundant_claim(claim, validated_claims):
                continue
                
            validated_claims.append(claim)
        
        # 7. Validate comprehensive coverage (address omitted claims)
        validated_claims = self._validate_comprehensive_coverage(validated_claims, sentence)
        
        return validated_claims

    def _is_unverifiable_claim(self, claim: str) -> bool:
        """Check if claim is unverifiable (personal narratives, patient interactions)"""
        unverifiable_patterns = [
            r'\b(I|you|your|we|us|our)\b.*(spoke|talked|discussed|mentioned)',
            r'\b(you|your)\b.*(are|were|will be)\b.*(experiencing|feeling|having)',
            r'\b(pain|discomfort|symptoms)\b.*\b(very|extremely|quite)\b.*\b(tiring|difficult|challenging)\b',
            r'\b(I|we|us)\b.*\b(recommend|suggest|think|believe)\b',
            r'\b(your doctor|your partner|your family)\b',
            r'\b(personal|individual|specific)\b.*\b(experience|situation|case)\b'
        ]
        
        claim_lower = claim.lower()
        for pattern in unverifiable_patterns:
            if re.search(pattern, claim_lower):
                return True
        return False

    def _is_grounded_in_sentence(self, claim: str, sentence: str) -> bool:
        """Check if claim is grounded in the original sentence"""
        # Extract key medical terms and concepts from sentence
        sentence_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
        claim_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))
        
        # Check if claim shares significant terms with sentence
        overlap = len(sentence_terms.intersection(claim_terms))
        total_claim_terms = len(claim_terms)
        
        if total_claim_terms == 0:
            return False
            
        # At least 30% of claim terms should be from the sentence
        return overlap / total_claim_terms >= 0.3

    def _is_incomplete_claim(self, claim: str, sentence: str) -> bool:
        """Check if claim is incomplete (missing important modifiers)"""
        # Look for important modifiers in sentence that might be missing in claim
        important_modifiers = [
            r'\b(while|although|however|but|yet)\b',
            r'\b(may|might|could|should|would)\b',
            r'\b(typically|usually|often|sometimes|rarely)\b',
            r'\b(significant|major|minor|serious|mild)\b',
            r'\b(positive|negative|adverse|beneficial)\b'
        ]
        
        sentence_has_modifiers = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in important_modifiers)
        claim_has_modifiers = any(re.search(pattern, claim, re.IGNORECASE) for pattern in important_modifiers)
        
        # If sentence has important modifiers but claim doesn't, it might be incomplete
        return sentence_has_modifiers and not claim_has_modifiers

    def _transform_to_declarative(self, claim: str) -> str:
        """Transform imperative or nested claims to declarative format"""
        # Handle imperative statements
        if claim.strip().endswith('.'):
            claim = claim.strip()[:-1]
            
        # Transform imperatives to declaratives
        imperative_patterns = [
            (r'^Take\s+(.+)$', r'Taking \1 is helpful for certain conditions'),
            (r'^Use\s+(.+)$', r'Using \1 is beneficial for certain conditions'),
            (r'^Apply\s+(.+)$', r'Applying \1 is effective for certain conditions'),
            (r'^Follow\s+(.+)$', r'Following \1 is recommended for certain conditions'),
        ]
        
        for pattern, replacement in imperative_patterns:
            if re.match(pattern, claim, re.IGNORECASE):
                claim = re.sub(pattern, replacement, claim, flags=re.IGNORECASE)
                break
        
        # Remove nested sub-clauses
        claim = re.sub(r'^(They|The doctor|Your doctor)\s+(said|mentioned|noted|explained)\s+that\s+', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'^(They|The doctor|Your doctor)\s+(said|mentioned|noted|explained)\s+', '', claim, flags=re.IGNORECASE)
        
        return claim

    def _resolve_context_dependencies(self, claim: str, context: str) -> str:
        """Resolve vague references and pronouns using context"""
        # Replace common pronouns with context-appropriate terms
        replacements = {
            r'\bthis\b': 'the mentioned',
            r'\bthat\b': 'the mentioned',
            r'\bthese\b': 'the mentioned',
            r'\bthose\b': 'the mentioned',
            r'\bit\b': 'the condition',
            r'\bthey\b': 'medical professionals',
            r'\bthem\b': 'medical professionals',
        }
        
        for pattern, replacement in replacements.items():
            claim = re.sub(pattern, replacement, claim, flags=re.IGNORECASE)
        
        # Extract specific entities from context to replace vague references
        # Look for medical entities in context
        medical_entities = re.findall(r'\b(doctor|physician|specialist|patient|medication|treatment|condition|disease|symptom)\b', context, re.IGNORECASE)
        if medical_entities:
            # Replace generic references with specific entities when appropriate
            if 'the mentioned' in claim.lower() and medical_entities:
                claim = claim.replace('the mentioned', medical_entities[0])
        
        return claim

    def _is_redundant_claim(self, claim: str, existing_claims: List[str]) -> bool:
        """Check if claim is redundant with existing claims"""
        if not existing_claims:
            return False
            
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        
        for existing_claim in existing_claims:
            existing_words = set(re.findall(r'\b\w+\b', existing_claim.lower()))
            
            # Calculate similarity based on word overlap
            if len(claim_words) > 0 and len(existing_words) > 0:
                overlap = len(claim_words.intersection(existing_words))
                similarity = overlap / max(len(claim_words), len(existing_words))
                
                if similarity >= self.similarity_threshold:
                    return True
        
        return False

    def _validate_comprehensive_coverage(self, claims: List[str], sentence: str) -> List[str]:
        """Validate that all important medical information is covered (7th taxonomy issue)"""
        if not claims:
            return claims
            
        # Extract key medical concepts from sentence
        medical_concepts = self._extract_medical_concepts(sentence)
        
        # Check if all important concepts are covered
        covered_concepts = set()
        for claim in claims:
            claim_concepts = self._extract_medical_concepts(claim)
            covered_concepts.update(claim_concepts)
        
        # Find missing concepts
        missing_concepts = medical_concepts - covered_concepts
        
        # If significant concepts are missing, add them as additional claims
        additional_claims = []
        for concept in missing_concepts:
            if len(concept.split()) >= 2:  # Only add substantial concepts
                additional_claims.append(f"The sentence mentions {concept}.")
        
        return claims + additional_claims

    def _extract_medical_concepts(self, text: str) -> set:
        """Extract medical concepts from text"""
        # Common medical terms and patterns
        medical_patterns = [
            r'\b(medication|drug|treatment|therapy|surgery|procedure)\b',
            r'\b(condition|disease|disorder|syndrome|illness)\b',
            r'\b(symptom|sign|indication|manifestation)\b',
            r'\b(diagnosis|prognosis|outcome|result)\b',
            r'\b(patient|doctor|physician|specialist|nurse)\b',
            r'\b(hospital|clinic|medical|healthcare)\b',
            r'\b(risk|benefit|side effect|complication)\b',
            r'\b(dose|dosage|frequency|duration)\b'
        ]
        
        concepts = set()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(matches)
        
        # Also extract noun phrases that might be medical concepts
        noun_phrases = re.findall(r'\b\w+\s+\w+\b', text)
        medical_noun_phrases = [phrase for phrase in noun_phrases 
                              if any(term in phrase.lower() for term in 
                                   ['health', 'medical', 'treatment', 'condition', 'symptom', 'disease'])]
        concepts.update(medical_noun_phrases)
        
        return concepts
