import re
from typing import List, Dict, Any

from .llm_provider import LLMProvider


class ChainOfThoughtReasoner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def reason_medical_atomic_facts_decompose(self):
        return {
            "reasoning_steps": [
                # Step 1: Identify main medical concepts
                "1. IDENTIFY MAIN MEDICAL CONCEPTS: First, identify the main medical concepts in the sentence - conditions, treatments, medications, procedures, or symptoms",

                # Step 2: Break down into verifiable facts
                "2. BREAK DOWN INTO VERIFIABLE FACTS: Then, break down each concept into verifiable facts that can be independently assessed",

                # Step 3: Ensure objectivity and verifiability
                "3. ENSURE OBJECTIVITY AND VERIFIABILITY: Next, ensure each fact is objective and can be verified against reliable medical sources",

                # Step 4: Validate completeness and context
                "4. VALIDATE COMPLETENESS AND CONTEXT: Then, validate that each fact is complete and contextually appropriate for medical assessment",

                # Step 5: MedScore Problem 1 - Filter unverifiable claims
                "5. FILTER UNVERIFIABLE CLAIMS: Remove personal narratives, patient-specific interactions, and bedside manner statements. Exclude: personal experiences ('I spoke with your doctor'), patient interactions, and subjective statements",

                # Step 6: MedScore Problem 2 - Prevent hallucinated claims
                "6. PREVENT HALLUCINATED CLAIMS: Ensure all claims are grounded in the original sentence. Verify no additional information beyond the sentence, no distortion of original meaning, and no irrelevant information",

                # Step 7: MedScore Problem 3 - Preserve complete claims
                "7. PRESERVE COMPLETE CLAIMS: Maintain important modifiers, conditional statements, temporal information, and cause-effect relationships. Ensure no critical medical details are lost",

                # Step 8: MedScore Problem 4 - Transform to declarative format
                "8. TRANSFORM TO DECLARATIVE FORMAT: Convert imperatives to declaratives, remove nested sub-clauses, and ensure declarative sentence structure. Transform reported speech into factual statements (e.g., 'They said/believed/mentioned that [claim]' to '[claim]')",

                # Step 9: MedScore Problem 5 - Resolve context-dependent claims
                "9. RESOLVE CONTEXT-DEPENDENT CLAIMS: Replace ALL pronouns with specific entities from context: 'it/this/that/these/those' → specific medical terms, 'his/her/your/their' → specific persons, 'they/them' → specific medical professionals",

                # Step 10: MedScore Problem 6 - Eliminate redundant claims
                "10. ELIMINATE REDUNDANT CLAIMS: Remove duplicate or minimally different versions of the same fact. Focus on distinct, non-overlapping medical facts",

                # Step 11: MedScore Problem 7 - Ensure comprehensive coverage
                "11. ENSURE COMPREHENSIVE COVERAGE: Verify all important medical information is extracted. Don't miss key facts from the sentence. Maintain completeness of medical content",

                # Step 12: Review and refine for accuracy
                "12. REVIEW AND REFINE FOR ACCURACY: Review each extracted fact for: objectivity, verifiability, completeness, accuracy, and medical relevance. Ensure each fact can stand alone as an independent medical statement",
            ],
        }

    def reason(
            self,
            query: str,
            evidence: list,
            temperature: float = 0.7
    ) -> Dict[str, Any]:
        prompt = self._build_cot_prompt(query, evidence)
        response = self.llm.generate(prompt, temperature=temperature, max_tokens=2000)

        print('===============[Cot Response]\n', response)

        # Parse the response into reasoning steps only
        steps = self._parse_reasoning_steps(response)

        # The reasoning stage focuses only on the reasoning process
        # Final answers will be generated in later pipeline stages
        return {
            "query": query,
            "reasoning_steps": steps,
            "raw_output": response
        }

    def _build_cot_prompt(self, query: str, evidence: list) -> str:
        evidence_text = ""
        if evidence:
            evidence_text = "\n".join([f"{e.content}" for e in evidence])
        prompt = f"""You are an expert logical reasoning assistant. Analyze the following problem systematically and provide detailed step-by-step reasoning.

{f'Evidence: {evidence_text}' if evidence_text else ''}

Problem: {query}

Instructions:
1. Read the problem carefully and identify all given information
2. Break down the problem into clear, logical reasoning steps
3. For each step, explain your thinking process clearly
4. Consider all possible scenarios and eliminate impossible ones
5. Use logical deduction and inference rules
6. For multiple choice questions, evaluate each option systematically
7. Show your work and reasoning for each step
8. Do NOT provide a final answer - focus only on the reasoning process

IMPORTANT: For logical reasoning problems:
- Identify all given conditions and constraints
- Use logical operators (if-then, and, or, not) correctly
- Consider all possible combinations and eliminate contradictions
- For spatial reasoning, visualize or map out the relationships
- For conditional logic, trace through all possible scenarios
- For argument analysis, identify premises, conclusions, and logical connections

Format your response as:

Step 1: [Your first reasoning step]
Step 2: [Your second reasoning step]
...

Begin your systematic reasoning:"""

        return prompt

    def _parse_reasoning_steps(self, response: str) -> List[str]:
        steps = []

        # Look for "Step X:" patterns
        step_pattern = r'Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|Final Answer:|$)'
        matches = re.finditer(step_pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_content = match.group(1).strip()
            if step_content:
                steps.append(step_content)

        return steps
