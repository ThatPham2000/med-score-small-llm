import asyncio
from typing import List, Dict, Any

from tqdm import tqdm

from llm import LLM
from utils import chunker
from verifier import Verifier


class VerifierInternalSmallLLM(Verifier):
    """
    Enhanced verifier for small language models using internal knowledge.
    This implementation makes small language models more intelligent by:
    1. Using chain-of-thought reasoning for verification
    2. Implementing multi-step fact-checking process
    3. Adding confidence scoring with threshold filtering
    4. Using structured reasoning prompts
    5. Filtering low-confidence verifications based on confidence_threshold
    """

    def __init__(
            self,
            llm: LLM = None,
            confidence_threshold: float = 0.7,
            reasoning_steps: int = 3,
    ):
        super().__init__(llm=llm)
        self.confidence_threshold = confidence_threshold
        self.reasoning_steps = reasoning_steps

    def add_evidence_to_verification_input(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add evidence field to verification input (internal knowledge mode)"""
        verification_input = []
        for d in decompositions:
            d["evidence"] = None  # Internal knowledge mode doesn't use external evidence
            verification_input.append(d)
        return verification_input

    def prepare_messages(self, verification_input: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Prepare messages with enhanced reasoning prompts for small LLMs using internal knowledge"""
        messages = []
        for d in verification_input:
            formatted_input = self._get_enhanced_verification_prompt(d['claim'])
            system_prompt = self._get_enhanced_system_prompt()

            messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_input}
            ])
        return messages

    def do_verify(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Override do_verify to use format_completions for enhanced processing"""
        verifier_inputs = self.add_evidence_to_verification_input(decompositions)
        messages = self.prepare_messages(verifier_inputs)

        all_completions = []
        n_iter = len(messages) // self.batch_size
        for batch in tqdm(chunker(messages, self.batch_size), desc="Verifier process", total=n_iter):
            completions = asyncio.run(self.llm.batch_response(batch))
            all_completions.extend(completions)

        # Use format_completions instead of basic parsing
        verification_output = self.format_completions(
            verifier_inputs,
            self.llm.normalize_llm_response(all_completions)
        )
        return verification_output

    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt with reasoning for small language models using internal knowledge"""
        reasoning_steps_text = self._generate_verification_reasoning_steps()
        reasoning_format = self._generate_verification_reasoning_format()

        return f"""You are an assistant who verifies whether a claim from a medical response is True or False using step-by-step reasoning and your own knowledge. 

REASONING PROCESS:
{reasoning_steps_text}

You should rely exclusively on your own knowledge and always output 'True' or 'False' first, followed by your reasoning. If there is not enough context or you are unable to verify the claim, then output 'False'.

REASONING FORMAT:
Think step by step:
{reasoning_format}

CONFIDENCE SCORING:
After your reasoning, also provide a confidence score from 0.0 to 1.0 indicating how certain you are about your verification.

Output: [True/False] - [Brief reasoning] - [Confidence: X.X]"""

    def _get_enhanced_verification_prompt(self, claim: str) -> str:
        """Enhanced verification prompt with reasoning steps for internal knowledge"""
        reasoning_format = self._generate_verification_reasoning_format()
        return f"""Please verify the following medical claim using step-by-step reasoning and your own knowledge:

Claim: {claim}

Think step by step:
{reasoning_format}

After your reasoning, also provide a confidence score from 0.0 to 1.0 indicating how certain you are about your verification.

Output: [True/False] - [Brief reasoning explaining your decision] - [Confidence: X.X]"""

    def format_completions(self, verifier_inputs: List[Dict[str, Any]], completions: List[str]) -> List[
        Dict[str, Any]]:
        """Enhanced completion formatting that handles reasoning, confidence, and threshold filtering"""
        verifications = []
        for verifier_input, completion in zip(verifier_inputs, completions):
            # Extract True/False, reasoning, and confidence from completion
            raw_response, score, confidence = self._parse_reasoning_response_with_confidence(completion)

            # Apply confidence threshold filtering
            if confidence < self.confidence_threshold:
                # If confidence is below threshold, mark as uncertain
                score = 0.0  # Treat low-confidence verifications as False
                raw_response = f"[LOW CONFIDENCE] {raw_response}"

            verification = {k: v for k, v in verifier_input.items()}
            verification["raw"] = raw_response
            verification["score"] = score
            verification["confidence"] = confidence
            verification["meets_threshold"] = confidence >= self.confidence_threshold
            verifications.append(verification)

        return verifications

    def _parse_reasoning_response_with_confidence(self, completion: str) -> tuple:
        """Parse reasoning response to extract True/False, score, and confidence"""
        lines = completion.strip().split('\n')

        # Initialize default values
        raw_response = "False"
        score = 0.0
        confidence = 0.5  # Default confidence

        # Look for True/False in the response
        for line in lines:
            line = line.strip()
            if line.lower().startswith('true'):
                raw_response = "True"
                score = 1.0
                break
            elif line.lower().startswith('false'):
                raw_response = "False"
                score = 0.0
                break

        # If no clear True/False found, try to infer from content
        if raw_response == "False" and score == 0.0:
            completion_lower = completion.lower()
            if any(word in completion_lower for word in ['true', 'correct', 'accurate', 'valid']):
                raw_response = "True"
                score = 1.0
            elif any(word in completion_lower for word in ['false', 'incorrect', 'inaccurate', 'invalid']):
                raw_response = "False"
                score = 0.0

        # Extract confidence score from the response
        confidence = self._extract_confidence_score(completion)

        return raw_response, score, confidence

    def _extract_confidence_score(self, completion: str) -> float:
        """Extract confidence score from completion text"""
        import re

        # Look for confidence patterns like "Confidence: 0.8" or "[Confidence: 0.8]"
        confidence_patterns = [
            r'confidence:\s*(\d+\.?\d*)',
            r'\[confidence:\s*(\d+\.?\d*)\]',
            r'confidence\s*=\s*(\d+\.?\d*)',
            r'confidence\s*(\d+\.?\d*)',
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, completion.lower())
            if match:
                try:
                    confidence = float(match.group(1))
                    # Ensure confidence is between 0.0 and 1.0
                    return max(0.0, min(1.0, confidence))
                except ValueError:
                    continue

        # Look for percentage patterns like "80%" or "80 percent"
        percentage_patterns = [
            r'(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*percent',
        ]

        for pattern in percentage_patterns:
            match = re.search(pattern, completion.lower())
            if match:
                try:
                    percentage = float(match.group(1))
                    return max(0.0, min(1.0, percentage / 100.0))
                except ValueError:
                    continue

        # Look for word-based confidence indicators
        completion_lower = completion.lower()
        if any(word in completion_lower for word in ['very confident', 'highly confident', 'extremely confident']):
            return 0.9
        elif any(word in completion_lower for word in ['confident', 'certain', 'sure']):
            return 0.8
        elif any(word in completion_lower for word in ['somewhat confident', 'moderately confident']):
            return 0.6
        elif any(word in completion_lower for word in ['uncertain', 'unsure', 'not sure']):
            return 0.3
        elif any(word in completion_lower for word in ['very uncertain', 'highly uncertain']):
            return 0.1

        # Default confidence based on response clarity
        if any(word in completion_lower for word in ['true', 'false', 'correct', 'incorrect']):
            return 0.7  # Medium confidence for clear responses
        else:
            return 0.4  # Low confidence for unclear responses

    def _generate_verification_reasoning_steps(self) -> str:
        """Generate dynamic reasoning steps for verification based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. Determine if the claim is factually correct based on your medical knowledge"
        elif self.reasoning_steps == 2:
            return """1. First, identify the key medical concepts in the claim
2. Then, determine if the claim is factually correct based on your knowledge"""
        elif self.reasoning_steps == 3:
            return """1. First, identify the key medical concepts in the claim
2. Then, recall your knowledge about these concepts
3. Finally, determine if the claim is factually correct"""
        elif self.reasoning_steps == 4:
            return """1. First, identify the key medical concepts in the claim
2. Then, recall your knowledge about these concepts
3. Next, evaluate the accuracy of the claim
4. Finally, determine if the claim is factually correct"""
        elif self.reasoning_steps == 5:
            return """1. First, identify the key medical concepts in the claim
2. Then, recall your knowledge about these concepts
3. Next, evaluate the accuracy of the claim
4. Then, consider any potential ambiguities or edge cases
5. Finally, determine if the claim is factually correct"""
        else:
            # Default to 3 steps for any other value
            return """1. First, identify the key medical concepts in the claim
2. Then, recall your knowledge about these concepts
3. Finally, determine if the claim is factually correct"""

    def _generate_verification_reasoning_format(self) -> str:
        """Generate dynamic reasoning format for verification based on the reasoning_steps parameter"""
        if self.reasoning_steps == 1:
            return "1. Is this claim factually correct based on my medical knowledge?"
        elif self.reasoning_steps == 2:
            return """1. What medical concepts are mentioned in this claim?
2. Is this claim factually correct based on my knowledge?"""
        elif self.reasoning_steps == 3:
            return """1. What medical concepts are mentioned in this claim?
2. What do I know about these concepts from medical knowledge?
3. Is this claim factually correct based on my knowledge?"""
        elif self.reasoning_steps == 4:
            return """1. What medical concepts are mentioned in this claim?
2. What do I know about these concepts from medical knowledge?
3. How accurate is this claim based on my knowledge?
4. Is this claim factually correct?"""
        elif self.reasoning_steps == 5:
            return """1. What medical concepts are mentioned in this claim?
2. What do I know about these concepts from medical knowledge?
3. How accurate is this claim based on my knowledge?
4. Are there any potential ambiguities or edge cases?
5. Is this claim factually correct?"""
        else:
            # Default to 3 steps for any other value
            return """1. What medical concepts are mentioned in this claim?
2. What do I know about these concepts from medical knowledge?
3. Is this claim factually correct based on my knowledge?"""

    def get_confidence_statistics(self, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get confidence statistics from verification results"""
        if not verifications:
            return {
                "total_verifications": 0,
                "average_confidence": 0.0,
                "confidence_distribution": {},
                "threshold_meeting_rate": 0.0,
                "low_confidence_count": 0
            }

        confidences = [v.get("confidence", 0.0) for v in verifications]
        meets_threshold = [v.get("meets_threshold", False) for v in verifications]

        # Calculate statistics
        total_verifications = len(verifications)
        average_confidence = sum(confidences) / total_verifications if total_verifications > 0 else 0.0
        threshold_meeting_rate = sum(meets_threshold) / total_verifications if total_verifications > 0 else 0.0
        low_confidence_count = sum(1 for c in confidences if c < self.confidence_threshold)

        # Confidence distribution
        confidence_ranges = {
            "very_high": sum(1 for c in confidences if c >= 0.9),
            "high": sum(1 for c in confidences if 0.7 <= c < 0.9),
            "medium": sum(1 for c in confidences if 0.5 <= c < 0.7),
            "low": sum(1 for c in confidences if 0.3 <= c < 0.5),
            "very_low": sum(1 for c in confidences if c < 0.3)
        }

        return {
            "total_verifications": total_verifications,
            "average_confidence": round(average_confidence, 3),
            "confidence_threshold": self.confidence_threshold,
            "threshold_meeting_rate": round(threshold_meeting_rate, 3),
            "low_confidence_count": low_confidence_count,
            "confidence_distribution": confidence_ranges,
            "confidence_range": {
                "min": round(min(confidences), 3) if confidences else 0.0,
                "max": round(max(confidences), 3) if confidences else 0.0
            }
        }

    def filter_by_confidence(self, verifications: List[Dict[str, Any]],
                             min_confidence: float = None) -> List[Dict[str, Any]]:
        """Filter verifications by confidence threshold"""
        if min_confidence is None:
            min_confidence = self.confidence_threshold

        return [v for v in verifications if v.get("confidence", 0.0) >= min_confidence]
