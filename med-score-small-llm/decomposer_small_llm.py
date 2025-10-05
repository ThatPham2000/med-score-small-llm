import re
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            adaptive_thresholds: bool = True,
    ):
        super().__init__(llm=llm)
        self.reasoning_steps = reasoning_steps
        self.enable_validation = enable_validation
        self.similarity_threshold = similarity_threshold
        self.adaptive_thresholds = adaptive_thresholds

    def get_system_prompt(self) -> str:
        """Enhanced system prompt addressing the 7 MedScoreTaxonomy issues with improved small LLM guidance"""
        # Generate dynamic reasoning steps based on the reasoning_steps parameter
        reasoning_steps_text = self._generate_reasoning_steps()
        reasoning_format = self._generate_reasoning_format()

        # Generate dynamic examples based on reasoning steps
        examples = self._generate_examples()

        return f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning while addressing the 7 MedScoreTaxonomy issues.

CRITICAL INSTRUCTIONS FOR SMALL LLMs:
- Focus on ONE medical concept per claim
- Use simple, declarative sentences
- Avoid complex medical jargon when possible
- Extract facts that can be verified independently
- Preserve all important medical modifiers and conditions

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
        """Enhanced input formatting with reasoning prompts optimized for small LLMs"""
        reasoning_format = self._generate_reasoning_format()
        return f"""Context: {context}

Please breakdown the following sentence into independent facts: {sentence}

IMPORTANT: Extract ONE medical concept per claim. Use simple, declarative sentences.

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
                claim_list = self._validate_and_filter_claims(claim_list, d_input.get("sentence", ""),
                                                              d_input.get("context", ""))

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

    def _calculate_adaptive_threshold(self, sentence: str) -> float:
        """Calculate adaptive similarity threshold based on sentence complexity"""
        if not self.adaptive_thresholds:
            return self.similarity_threshold

        # Calculate sentence complexity factors
        word_count = len(sentence.split())
        medical_terms = len(re.findall(
            r'\b(?:patient|doctor|medical|treatment|condition|disease|symptom|diagnosis|therapy|medication|surgery|procedure)\b',
            sentence.lower()))
        conditional_words = len(
            re.findall(r'\b(?:if|when|unless|provided|while|although|however|but|yet|despite|whereas)\b',
                       sentence.lower()))

        # Adjust threshold based on complexity
        base_threshold = self.similarity_threshold

        # More complex sentences need lower thresholds (more lenient)
        if word_count > 20:
            base_threshold -= 0.1
        if medical_terms > 3:
            base_threshold -= 0.05
        if conditional_words > 2:
            base_threshold -= 0.1

        # Ensure threshold stays within reasonable bounds
        return max(0.3, min(0.9, base_threshold))

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
        """Generate dynamic reasoning steps based on the reasoning_steps parameter with enhanced small LLM guidance"""
        if self.reasoning_steps == 1:
            return "1. Identify and extract all verifiable medical facts from the sentence (Focus on ONE concept per claim)"
        elif self.reasoning_steps == 2:
            return """1. First, identify the main medical concepts in the sentence (Identify 2-3 key concepts)
2. Then, extract verifiable facts from each concept (One simple fact per concept)"""
        elif self.reasoning_steps == 3:
            return """1. First, identify the main medical concepts in the sentence (Identify key medical terms)
2. Then, break down each concept into verifiable facts (Keep facts simple and declarative)
3. Finally, ensure each fact is objective and can be verified against reliable sources (Check for "if", "when", "typically", etc.)"""
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
            return """1. First, identify the main medical concepts in the sentence (Identify key medical terms)
2. Then, break down each concept into verifiable facts (Keep facts simple and declarative)
3. Finally, ensure each fact is objective and can be verified against reliable sources (Check for "if", "when", "typically", etc.)"""

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
            if self._is_redundant_claim(claim, validated_claims, sentence):
                continue

            validated_claims.append(claim)

        # 7. Validate comprehensive coverage (address omitted claims)
        validated_claims = self._validate_comprehensive_coverage(validated_claims, sentence)

        return validated_claims

    def _is_unverifiable_claim(self, claim: str) -> bool:
        """Check if claim is unverifiable using semantic analysis of personal narratives"""
        # Define semantic categories for unverifiable content based on MEDSCORE_PROMPT patterns
        unverifiable_indicators = {
            'personal_pronouns': ['i', 'you', 'your', 'we', 'us', 'our', 'my', 'mine'],
            'interaction_verbs': ['spoke', 'talked', 'discussed', 'mentioned', 'said', 'told', 'asked', 'wanted',
                                  'addressed'],
            'subjective_experiences': ['experiencing', 'feeling', 'having', 'going through', 'dealing with',
                                       'recovering'],
            'bedside_manner': ['tiring', 'difficult', 'challenging', 'frustrating', 'concerning', 'anxiety', 'worry'],
            'personal_references': ['your doctor', 'your partner', 'your family', 'your situation', 'your concerns'],
            'subjective_opinions': ['recommend', 'suggest', 'think', 'believe', 'feel', 'consider', 'advise'],
            'patient_specific': ['your', 'you', 'yourself', 'your body', 'your condition', 'your health']
        }

        claim_lower = claim.lower()

        # Check for multiple indicators that suggest unverifiable content
        indicator_count = 0

        # Check for personal pronouns combined with interaction verbs (strong indicator)
        has_personal_pronoun = any(pronoun in claim_lower for pronoun in unverifiable_indicators['personal_pronouns'])
        has_interaction_verb = any(verb in claim_lower for verb in unverifiable_indicators['interaction_verbs'])

        if has_personal_pronoun and has_interaction_verb:
            indicator_count += 3  # Strong indicator

        # Check for subjective experiences
        if any(exp in claim_lower for exp in unverifiable_indicators['subjective_experiences']):
            indicator_count += 2

        # Check for bedside manner language
        if any(manner in claim_lower for manner in unverifiable_indicators['bedside_manner']):
            indicator_count += 2

        # Check for personal references
        if any(ref in claim_lower for ref in unverifiable_indicators['personal_references']):
            indicator_count += 2

        # Check for subjective opinions
        if any(opinion in claim_lower for opinion in unverifiable_indicators['subjective_opinions']):
            indicator_count += 1

        # Check for patient-specific language
        if any(patient in claim_lower for patient in unverifiable_indicators['patient_specific']):
            indicator_count += 1

        # If multiple indicators are present, consider it unverifiable
        return indicator_count >= 2

    def _is_grounded_in_sentence(self, claim: str, sentence: str) -> bool:
        """Check if claim is grounded in the original sentence using semantic similarity"""
        try:
            # Use TF-IDF vectorization for semantic similarity
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            texts = [sentence, claim]
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Also check for key medical terms overlap as a fallback
            sentence_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
            claim_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))

            if len(claim_terms) > 0:
                term_overlap = len(sentence_terms.intersection(claim_terms)) / len(claim_terms)
            else:
                term_overlap = 0

            # Use either semantic similarity or term overlap
            return similarity >= 0.2 or term_overlap >= 0.3

        except Exception:
            # Fallback to simple term overlap if TF-IDF fails
            sentence_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
            claim_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))

            if len(claim_terms) == 0:
                return False

            overlap = len(sentence_terms.intersection(claim_terms))
            return overlap / len(claim_terms) >= 0.3

    def _is_incomplete_claim(self, claim: str, sentence: str) -> bool:
        """Check if claim is incomplete using semantic analysis of important concepts"""
        # Define categories of important modifiers and qualifiers based on MEDSCORE_PROMPT patterns
        important_concepts = {
            'conditional_words': ['while', 'although', 'however', 'but', 'yet', 'despite', 'whereas', 'if', 'when'],
            'uncertainty_modals': ['may', 'might', 'could', 'should', 'would', 'can', 'might be', 'possibly'],
            'frequency_adverbs': ['typically', 'usually', 'often', 'sometimes', 'rarely', 'commonly', 'frequently',
                                  'generally'],
            'intensity_modifiers': ['significant', 'major', 'minor', 'serious', 'mild', 'severe', 'substantial',
                                    'considerable'],
            'evaluation_terms': ['positive', 'negative', 'adverse', 'beneficial', 'harmful', 'effective', 'ineffective',
                                 'helpful'],
            'temporal_indicators': ['before', 'after', 'during', 'while', 'when', 'since', 'until', 'initially',
                                    'eventually'],
            'causal_indicators': ['because', 'due to', 'caused by', 'leads to', 'results in', 'contributes to',
                                  'in turn'],
            'medical_qualifiers': ['relatively', 'typically', 'usually', 'often', 'sometimes', 'rarely', 'commonly']
        }

        # Extract concepts from sentence and claim
        sentence_lower = sentence.lower()
        claim_lower = claim.lower()

        sentence_concepts = set()
        claim_concepts = set()

        for category, words in important_concepts.items():
            for word in words:
                if word in sentence_lower:
                    sentence_concepts.add(category)
                if word in claim_lower:
                    claim_concepts.add(category)

        # Check if sentence has important concepts that are missing from claim
        missing_concepts = sentence_concepts - claim_concepts

        # If significant concepts are missing, the claim might be incomplete
        return len(missing_concepts) >= 2 or (len(missing_concepts) >= 1 and len(sentence_concepts) >= 3)

    def _transform_to_declarative(self, claim: str) -> str:
        """Transform imperative or nested claims to declarative format using general patterns"""
        # Handle imperative statements
        if claim.strip().endswith('.'):
            claim = claim.strip()[:-1]

        # Define general imperative patterns and their declarative transformations
        imperative_verbs = [
            'take', 'use', 'apply', 'follow', 'avoid', 'consider', 'try', 'start', 'stop', 'continue',
            'begin', 'end', 'finish', 'complete', 'perform', 'conduct', 'administer', 'prescribe',
            'schedule', 'arrange', 'plan', 'prepare', 'monitor', 'check', 'review', 'examine'
        ]

        # Check if claim starts with an imperative verb
        claim_lower = claim.lower().strip()
        for verb in imperative_verbs:
            if claim_lower.startswith(verb + ' '):
                # Transform to declarative format
                rest_of_claim = claim[len(verb):].strip()
                claim = f"{verb.capitalize()}ing {rest_of_claim} is beneficial for certain conditions"
                break

        # Remove nested sub-clauses and reported speech based on MEDSCORE_PROMPT patterns
        report_verbs = ['said', 'mentioned', 'noted', 'explained', 'stated', 'indicated', 'reported', 'wanted',
                        'addressed']
        report_subjects = ['they', 'the doctor', 'your doctor', 'the physician', 'the specialist', 'doctors']

        for subject in report_subjects:
            for verb in report_verbs:
                # Pattern: "Subject verb that ..."
                pattern = rf'^{re.escape(subject)}\s+{re.escape(verb)}\s+that\s+'
                if re.match(pattern, claim, re.IGNORECASE):
                    claim = re.sub(pattern, '', claim, flags=re.IGNORECASE)
                    break
                # Pattern: "Subject verb ..."
                pattern = rf'^{re.escape(subject)}\s+{re.escape(verb)}\s+'
                if re.match(pattern, claim, re.IGNORECASE):
                    claim = re.sub(pattern, '', claim, flags=re.IGNORECASE)
                    break

        return claim

    def _resolve_context_dependencies(self, claim: str, context: str) -> str:
        """Resolve vague references and pronouns using context analysis"""
        # Extract entities from context
        context_entities = self._extract_entities_from_context(context)

        # Define vague reference patterns and their context-appropriate replacements
        vague_references = {
            'demonstrative_pronouns': ['this', 'that', 'these', 'those'],
            'personal_pronouns': ['it', 'they', 'them', 'their'],
            'temporal_references': ['now', 'then', 'recently', 'previously'],
            'spatial_references': ['here', 'there', 'this place', 'that location']
        }

        claim_lower = claim.lower()

        # Replace demonstrative pronouns with context-appropriate terms
        for pronoun in vague_references['demonstrative_pronouns']:
            if f' {pronoun} ' in f' {claim_lower} ':
                # Find the most relevant entity from context
                best_entity = self._find_best_context_entity(pronoun, context_entities, claim)
                if best_entity:
                    claim = re.sub(rf'\b{re.escape(pronoun)}\b', best_entity, claim, flags=re.IGNORECASE)
                else:
                    claim = re.sub(rf'\b{re.escape(pronoun)}\b', 'the mentioned', claim, flags=re.IGNORECASE)

        # Replace personal pronouns with context-appropriate terms
        for pronoun in vague_references['personal_pronouns']:
            if f' {pronoun} ' in f' {claim_lower} ':
                best_entity = self._find_best_context_entity(pronoun, context_entities, claim)
                if best_entity:
                    claim = re.sub(rf'\b{re.escape(pronoun)}\b', best_entity, claim, flags=re.IGNORECASE)
                else:
                    # Default replacements based on pronoun type
                    if pronoun in ['they', 'them', 'their']:
                        claim = re.sub(rf'\b{re.escape(pronoun)}\b', 'medical professionals', claim,
                                       flags=re.IGNORECASE)
                    elif pronoun == 'it':
                        claim = re.sub(rf'\b{re.escape(pronoun)}\b', 'the condition', claim, flags=re.IGNORECASE)

        return claim

    def _extract_entities_from_context(self, context: str) -> List[str]:
        """Extract relevant entities from context"""
        # Medical entity patterns based on MEDSCORE_PROMPT examples
        medical_patterns = [
            r'\b(doctor|physician|specialist|nurse|practitioner)\b',
            r'\b(patient|person|individual)\b',
            r'\b(medication|drug|medicine|treatment|therapy)\b',
            r'\b(condition|disease|disorder|syndrome|illness)\b',
            r'\b(symptom|sign|indication|manifestation)\b',
            r'\b(hospital|clinic|medical center|healthcare facility)\b',
            r'\b(anabolic steroids|substances|medications)\b',
            r'\b(muscle|bone|health|side effects|risks)\b'
        ]

        entities = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            entities.extend(matches)

        return list(set(entities))  # Remove duplicates

    def _find_best_context_entity(self, pronoun: str, entities: List[str], claim: str) -> str:
        """Find the best context entity to replace a pronoun"""
        if not entities:
            return None

        # Simple heuristic: find entity that appears in both context and claim
        claim_lower = claim.lower()
        for entity in entities:
            if entity.lower() in claim_lower:
                return entity

        # If no direct match, return the first relevant entity
        return entities[0] if entities else None

    def _is_redundant_claim(self, claim: str, existing_claims: List[str], sentence: str = "") -> bool:
        """Check if claim is redundant with existing claims using semantic similarity with adaptive thresholds"""
        if not existing_claims:
            return False

        # Use adaptive threshold if enabled
        threshold = self._calculate_adaptive_threshold(sentence) if sentence else self.similarity_threshold

        try:
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            all_texts = [claim] + existing_claims
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Calculate similarity between the new claim and existing claims
            claim_vector = tfidf_matrix[0:1]
            existing_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(claim_vector, existing_vectors)[0]

            # Check if any similarity exceeds adaptive threshold
            return any(sim >= threshold for sim in similarities)

        except Exception:
            # Fallback to simple word overlap
            claim_words = set(re.findall(r'\b\w+\b', claim.lower()))

            for existing_claim in existing_claims:
                existing_words = set(re.findall(r'\b\w+\b', existing_claim.lower()))

                if len(claim_words) > 0 and len(existing_words) > 0:
                    overlap = len(claim_words.intersection(existing_words))
                    similarity = overlap / max(len(claim_words), len(existing_words))

                    if similarity >= self.similarity_threshold:
                        return True

            return False

    def _validate_comprehensive_coverage(self, claims: List[str], sentence: str) -> List[str]:
        """Validate that all important medical information is covered using semantic analysis"""
        if not claims:
            return claims

        # Extract key medical concepts from sentence using multiple approaches
        sentence_concepts = self._extract_medical_concepts_comprehensive(sentence)

        # Check coverage using semantic similarity
        covered_concepts = set()
        for claim in claims:
            claim_concepts = self._extract_medical_concepts_comprehensive(claim)
            covered_concepts.update(claim_concepts)

        # Find missing concepts using semantic similarity
        missing_concepts = self._find_missing_concepts_semantic(sentence_concepts, covered_concepts, claims, sentence)

        # Add missing concepts as additional claims
        additional_claims = []
        for concept in missing_concepts:
            if len(concept.split()) >= 2:  # Only add substantial concepts
                additional_claims.append(f"The sentence mentions {concept}.")

        return claims + additional_claims

    def _extract_medical_concepts_comprehensive(self, text: str) -> set:
        """Extract medical concepts from text using comprehensive approach"""
        # Medical domain categories based on MEDSCORE_PROMPT patterns
        medical_categories = {
            'medical_procedures': ['treatment', 'therapy', 'surgery', 'procedure', 'intervention', 'operation', 'shot',
                                   'vaccine'],
            'medical_conditions': ['condition', 'disease', 'disorder', 'syndrome', 'illness', 'pathology', 'infection'],
            'symptoms': ['symptom', 'sign', 'indication', 'manifestation', 'presentation', 'soreness', 'pain'],
            'outcomes': ['diagnosis', 'prognosis', 'outcome', 'result', 'effect', 'consequence', 'complication'],
            'people': ['patient', 'doctor', 'physician', 'specialist', 'nurse', 'practitioner'],
            'facilities': ['hospital', 'clinic', 'medical', 'healthcare', 'facility'],
            'risks_benefits': ['risk', 'benefit', 'side effect', 'complication', 'adverse effect', 'positive effects'],
            'dosage': ['dose', 'dosage', 'frequency', 'duration', 'administration'],
            'substances': ['medication', 'drug', 'medicine', 'substances', 'steroids', 'anabolic steroids']
        }

        concepts = set()
        text_lower = text.lower()

        # Extract concepts by category
        for category, terms in medical_categories.items():
            for term in terms:
                if term in text_lower:
                    concepts.add(term)

        # Extract medical noun phrases
        noun_phrases = re.findall(r'\b\w+\s+\w+\b', text)
        medical_indicators = ['health', 'medical', 'treatment', 'condition', 'symptom', 'disease', 'therapy',
                              'clinical']

        for phrase in noun_phrases:
            if any(indicator in phrase.lower() for indicator in medical_indicators):
                concepts.add(phrase)

        # Extract specific medical terms using patterns
        medical_patterns = [
            r'\b\w+\s+(disease|syndrome|disorder|condition|treatment|therapy)\b',
            r'\b(medical|clinical|healthcare|therapeutic)\s+\w+\b',
            r'\b\w+\s+(medication|drug|medicine|treatment)\b',
            r'\b(anabolic|steroids|substances|medications)\b',
            r'\b(muscle|bone|health|side effects|risks)\b'
        ]

        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(matches)

        return concepts

    def _find_missing_concepts_semantic(self, sentence_concepts: set, covered_concepts: set, claims: List[str],
                                        sentence: str) -> set:
        """Find missing concepts using semantic similarity"""
        missing_concepts = set()

        # Direct set difference for exact matches
        direct_missing = sentence_concepts - covered_concepts
        missing_concepts.update(direct_missing)

        # Use semantic similarity to find concepts that might be covered but with different wording
        try:
            if claims:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                all_texts = [sentence] + claims
                tfidf_matrix = vectorizer.fit_transform(all_texts)

                # Calculate similarity between sentence and claims
                sentence_vector = tfidf_matrix[0:1]
                claim_vectors = tfidf_matrix[1:]

                similarities = cosine_similarity(sentence_vector, claim_vectors)[0]

                # If overall similarity is low, there might be missing concepts
                if max(similarities) < 0.5:  # Low similarity threshold
                    # Add some key concepts that might be missing
                    for concept in sentence_concepts:
                        if concept not in covered_concepts:
                            missing_concepts.add(concept)

        except Exception:
            # Fallback to direct comparison
            missing_concepts.update(sentence_concepts - covered_concepts)

        return missing_concepts
