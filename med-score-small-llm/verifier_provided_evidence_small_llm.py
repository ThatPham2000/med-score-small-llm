# import asyncio
# from typing import List, Dict, Any
#
# from tqdm import tqdm
#
# from llm import LLM
# from utils import chunker, parse_reasoning_response_with_confidence
# from verifier import Verifier
#
#
# class VerifierProvidedEvidenceSmallLLM(Verifier):
#     """
#     Enhanced verifier for small language models using provided evidence.
#     This implementation makes small language models more intelligent by:
#     1. Using chain-of-thought reasoning for verification
#     2. Implementing multi-step fact-checking process with provided evidence
#     3. Adding confidence scoring with threshold filtering
#     4. Using structured reasoning prompts
#     5. Filtering low-confidence verifications based on confidence_threshold
#     """
#
#     def __init__(
#             self,
#             provided_evidence: Dict[str, str],
#             llm: LLM = None,
#             confidence_threshold: float = 0.7,
#             reasoning_steps: int = 3,
#     ):
#         super().__init__(llm=llm)
#         self.provided_evidence = provided_evidence
#         self.confidence_threshold = confidence_threshold
#         self.reasoning_steps = reasoning_steps
#
#     def add_evidence_to_verification_input(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Add evidence field to verification input using provided evidence"""
#         verification_input = []
#         for d in decompositions:
#             # Use provided evidence if available, otherwise None
#             evidence = self.provided_evidence.get(d.get("id", ""), "")
#             d["evidence"] = evidence
#             verification_input.append(d)
#         return verification_input
#
#     def prepare_messages(self, verification_input: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
#         """Prepare messages with enhanced reasoning prompts for small LLMs using provided evidence"""
#         messages = []
#         for d in verification_input:
#             formatted_input = self._get_enhanced_verification_prompt_with_evidence(d['claim'], d['evidence'])
#             system_prompt = self._get_enhanced_system_prompt()
#
#             messages.append([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": formatted_input}
#             ])
#         return messages
#
#     def do_verify(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Override do_verify to use format_completions for enhanced processing"""
#         verifier_inputs = self.add_evidence_to_verification_input(decompositions)
#         messages = self.prepare_messages(verifier_inputs)
#
#         all_completions = []
#         n_iter = len(messages) // self.batch_size
#         for batch in tqdm(chunker(messages, self.batch_size), desc="Verifier process", total=n_iter):
#             completions = asyncio.run(self.llm.batch_response(batch))
#             all_completions.extend(completions)
#
#         # Use format_completions instead of basic parsing
#         verification_output = self.format_completions(
#             verifier_inputs,
#             self.llm.normalize_llm_response(all_completions)
#         )
#         return verification_output
#
#     def _get_enhanced_system_prompt(self) -> str:
#         """Enhanced system prompt with reasoning for small language models using provided evidence only"""
#         reasoning_steps_text = self._generate_verification_reasoning_steps()
#         reasoning_format = self._generate_verification_reasoning_format()
#
#         return f"""You are an assistant who verifies whether a claim from a medical response is True or False using ONLY the provided evidence.
#
# REASONING PROCESS:
# {reasoning_steps_text}
#
# CRITICAL RULES:
# 1. You MUST ONLY use the provided evidence to verify claims
# 2. Do NOT use your own medical knowledge or training data
# 3. If the claim is not mentioned or supported by the provided evidence, it is considered a hallucination
# 4. If the claim contradicts the provided evidence, it is false
# 5. If the claim is not found in the evidence, output 'False' (hallucination)
# 6. Only output 'True' if the claim is explicitly supported by the provided evidence
#
# REASONING FORMAT:
# Think step by step:
# {reasoning_format}
#
# CONFIDENCE SCORING:
# After your reasoning, also provide a confidence score from 0.0 to 1.0 indicating how certain you are about your verification.
#
# Output: [True/False] - [Brief reasoning] - [Confidence: X.X]"""
#
#     def _get_enhanced_verification_prompt_with_evidence(self, claim: str, evidence: str) -> str:
#         """Enhanced verification prompt with reasoning steps and provided evidence only"""
#         reasoning_format = self._generate_verification_reasoning_format()
#         return f"""Please verify the following medical claim using ONLY the provided evidence. Do NOT use your own knowledge.
#
# Claim: {claim}
#
# Evidence: {evidence}
#
# IMPORTANT:
# - Only use information from the provided evidence above
# - If the claim is not mentioned in the evidence, it is a hallucination (False)
# - If the claim contradicts the evidence, it is False
# - Only mark as True if explicitly supported by the evidence
#
# Think step by step:
# {reasoning_format}
#
# After your reasoning, also provide a confidence score from 0.0 to 1.0 indicating how certain you are about your verification.
#
# Output: [True/False] - [Brief reasoning explaining your decision] - [Confidence: X.X]"""
#
#     def _get_system_prompt_for_evidence_mode(self, evidence: str = None) -> str:
#         """Get system prompt based on whether evidence is available"""
#         if evidence:
#             return """You are an assistant who verifies whether a claim from a medical response is True or False using provided evidence.
# You should rely on the provided evidence to make your determination. Always output 'True' or 'False' first, followed by your reasoning."""
#         else:
#             return """You are an assistant who verifies whether a claim from a medical response is True or False using your own knowledge.
# You should rely exclusively on your own knowledge and always output 'True' or 'False' first, followed by your reasoning."""
#
#     def format_completions(self, verifier_inputs: List[Dict[str, Any]], completions: List[str]) -> List[
#         Dict[str, Any]]:
#         """Enhanced completion formatting that handles reasoning, confidence, and threshold filtering"""
#         verifications = []
#         for verifier_input, completion in zip(verifier_inputs, completions):
#             # Extract True/False, reasoning, and confidence from completion
#             raw_response, score, confidence = parse_reasoning_response_with_confidence(completion)
#
#             # Apply confidence threshold filtering
#             if confidence < self.confidence_threshold:
#                 # If confidence is below threshold, mark as uncertain
#                 score = 0.0  # Treat low-confidence verifications as False
#                 raw_response = f"[LOW CONFIDENCE] [CONFIDENCE/THRESHOLD: {confidence}/{self.confidence_threshold}] {raw_response}"
#
#             verification = {k: v for k, v in verifier_input.items()}
#             verification["raw"] = raw_response
#             verification["score"] = score
#             verification["confidence"] = confidence
#             verification["meets_threshold"] = confidence >= self.confidence_threshold
#             verifications.append(verification)
#
#         return verifications
#
#     def _generate_verification_reasoning_steps(self) -> str:
#         """Generate dynamic reasoning steps for verification based on the reasoning_steps parameter"""
#         if self.reasoning_steps == 1:
#             return "1. Check if the claim is supported by the provided evidence only"
#         elif self.reasoning_steps == 2:
#             return """1. First, identify the key medical concepts in the claim
# 2. Then, check if these concepts are mentioned in the provided evidence"""
#         elif self.reasoning_steps == 3:
#             return """1. First, identify the key medical concepts in the claim
# 2. Then, search the provided evidence for information about these concepts
# 3. Finally, determine if the claim is supported by the evidence (if not found, it's a hallucination)"""
#         elif self.reasoning_steps == 4:
#             return """1. First, identify the key medical concepts in the claim
# 2. Then, search the provided evidence for information about these concepts
# 3. Next, check if the claim matches what is stated in the evidence
# 4. Finally, determine if the claim is supported by the evidence (if not found, it's a hallucination)"""
#         elif self.reasoning_steps == 5:
#             return """1. First, identify the key medical concepts in the claim
# 2. Then, search the provided evidence for information about these concepts
# 3. Next, check if the claim matches what is stated in the evidence
# 4. Then, verify there are no contradictions between the claim and evidence
# 5. Finally, determine if the claim is supported by the evidence (if not found, it's a hallucination)"""
#         else:
#             # Default to 3 steps for any other value
#             return """1. First, identify the key medical concepts in the claim
# 2. Then, search the provided evidence for information about these concepts
# 3. Finally, determine if the claim is supported by the evidence (if not found, it's a hallucination)"""
#
#     def _generate_verification_reasoning_format(self) -> str:
#         """Generate dynamic reasoning format for verification based on the reasoning_steps parameter"""
#         if self.reasoning_steps == 1:
#             return "1. Is this claim supported by the provided evidence only?"
#         elif self.reasoning_steps == 2:
#             return """1. What medical concepts are mentioned in this claim?
# 2. Are these concepts mentioned in the provided evidence?"""
#         elif self.reasoning_steps == 3:
#             return """1. What medical concepts are mentioned in this claim?
# 2. What does the evidence say about these concepts?
# 3. Is this claim supported by the evidence (if not found, it's a hallucination)?"""
#         elif self.reasoning_steps == 4:
#             return """1. What medical concepts are mentioned in this claim?
# 2. What does the evidence say about these concepts?
# 3. How well does the claim match what is stated in the evidence?
# 4. Is this claim supported by the evidence (if not found, it's a hallucination)?"""
#         elif self.reasoning_steps == 5:
#             return """1. What medical concepts are mentioned in this claim?
# 2. What does the evidence say about these concepts?
# 3. How well does the claim match what is stated in the evidence?
# 4. Are there any contradictions between the claim and evidence?
# 5. Is this claim supported by the evidence (if not found, it's a hallucination)?"""
#         else:
#             # Default to 3 steps for any other value
#             return """1. What medical concepts are mentioned in this claim?
# 2. What does the evidence say about these concepts?
# 3. Is this claim supported by the evidence (if not found, it's a hallucination)?"""
#
#     def get_confidence_statistics(self, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Get confidence statistics from verification results"""
#         if not verifications:
#             return {
#                 "total_verifications": 0,
#                 "average_confidence": 0.0,
#                 "confidence_distribution": {},
#                 "threshold_meeting_rate": 0.0,
#                 "low_confidence_count": 0
#             }
#
#         confidences = [v.get("confidence", 0.0) for v in verifications]
#         meets_threshold = [v.get("meets_threshold", False) for v in verifications]
#
#         # Calculate statistics
#         total_verifications = len(verifications)
#         average_confidence = sum(confidences) / total_verifications if total_verifications > 0 else 0.0
#         threshold_meeting_rate = sum(meets_threshold) / total_verifications if total_verifications > 0 else 0.0
#         low_confidence_count = sum(1 for c in confidences if c < self.confidence_threshold)
#
#         # Confidence distribution
#         confidence_ranges = {
#             "very_high": sum(1 for c in confidences if c >= 0.9),
#             "high": sum(1 for c in confidences if 0.7 <= c < 0.9),
#             "medium": sum(1 for c in confidences if 0.5 <= c < 0.7),
#             "low": sum(1 for c in confidences if 0.3 <= c < 0.5),
#             "very_low": sum(1 for c in confidences if c < 0.3)
#         }
#
#         return {
#             "total_verifications": total_verifications,
#             "average_confidence": round(average_confidence, 3),
#             "confidence_threshold": self.confidence_threshold,
#             "threshold_meeting_rate": round(threshold_meeting_rate, 3),
#             "low_confidence_count": low_confidence_count,
#             "confidence_distribution": confidence_ranges,
#             "confidence_range": {
#                 "min": round(min(confidences), 3) if confidences else 0.0,
#                 "max": round(max(confidences), 3) if confidences else 0.0
#             }
#         }
#
#     def filter_by_confidence(self, verifications: List[Dict[str, Any]],
#                              min_confidence: float = None) -> List[Dict[str, Any]]:
#         """Filter verifications by confidence threshold"""
#         if min_confidence is None:
#             min_confidence = self.confidence_threshold
#
#         return [v for v in verifications if v.get("confidence", 0.0) >= min_confidence]
