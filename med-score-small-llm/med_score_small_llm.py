import json
import os
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import jsonlines
import ollama

from decompose_dnd_score import DecomposeDnDScore
from decomposer_fact_score import DecomposerFactScore
from decomposer_med_score import DecomposerMedScore
from decomposer_small_llm import DecomposerSmallLLM
from exceptions import InvalidArgumentException, IllegalArgumentException
from llm_ollama import LLMOllama
from llm_openai import LLMOpenAI
from utils import parse_sentences
from verifier_internal import VerifierInternal
from verifier_provided_evidence import VerifierProvidedEvidence


def initialize_llm(llm_provider: str, model_name: str, server: Optional[str]):
    llm_provider = llm_provider.lower()
    if llm_provider == "ollama":
        if server is None:
            return LLMOllama(model_name=model_name)
        return LLMOllama(
            model_name=model_name,
            ollama_async_client=ollama.AsyncClient(host=server, verify=False)
        )

    if llm_provider == "openapi":
        if server is None:
            raise InvalidArgumentException("Server URL must be provided for OpenAPI LLM provider")
        return LLMOpenAI(model_name=model_name, server_path=server)

    raise IllegalArgumentException(f"Unknown LLM provider: {llm_provider}")


def initialize_decomposer(
        decomposition_mode: str,
        decomposition_llm_provider: str,
        decomposition_model_name: str,
        decomposition_server: Optional[str],
        reasoning_steps: int = 3,
):
    mode = decomposition_mode.lower()
    llm = initialize_llm(decomposition_llm_provider, decomposition_model_name, decomposition_server)

    if mode == "small_llm":
        return DecomposerSmallLLM(llm=llm)
    if mode == "medscore":
        return DecomposerMedScore(llm)
    if mode == "factscore":
        return DecomposerFactScore(llm)
    if mode == "dndscore":
        return DecomposeDnDScore(llm)
    else:
        raise IllegalArgumentException(f"Unknown decomposition mode: {mode}")


def initialize_verifier(
        verification_mode: str,
        verification_llm_provider: str,
        verification_model_name: str,
        verification_server: Optional[str],
        provided_evidence: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.7,
        reasoning_steps: int = 3,
):
    """Initialize verifier with multiple modes support"""
    mode = verification_mode.lower()
    llm = initialize_llm(verification_llm_provider, verification_model_name, verification_server)

    if mode == "internal":
        return VerifierInternal(llm)
    if mode == "provided":
        if provided_evidence is None:
            raise InvalidArgumentException("Provided evidence is required for 'provided' verification mode")
        return VerifierProvidedEvidence(provided_evidence, llm)
    # if mode == "internal_small_llm":
    #     return VerifierInternalSmallLLM(
    #         llm=llm,
    #         confidence_threshold=confidence_threshold,
    #         reasoning_steps=reasoning_steps
    #     )
    # if mode == "provided_small_llm":
    #     if provided_evidence is None:
    #         raise InvalidArgumentException("Provided evidence is required for 'provided_small_llm' verification mode")
    #     return VerifierProvidedEvidenceSmallLLM(
    #         provided_evidence=provided_evidence,
    #         llm=llm,
    #         confidence_threshold=confidence_threshold,
    #         reasoning_steps=reasoning_steps
    #     )

    raise IllegalArgumentException(f"Unknown verification mode: {mode}")


class MedScoreSmallLLM(object):
    """
    Enhanced MedScore implementation with multiple modes support.
    
    This class supports both traditional MedScore modes
    and enhanced small language model modes with:
    
    1. Chain-of-thought prompting
    2. Multi-step reasoning processes
    3. Enhanced decomposition with reasoning steps
    4. Improved verification with confidence scoring
    5. Context-aware fact extraction
    """

    def __init__(
            self,
            decomposition_mode: str = "small_llm",
            decomposition_llm_provider: str = "ollama",
            decomposition_model_name: str = "llama3.2:3b",
            decomposition_server: Optional[str] = None,
            decomposition_prompt_path: Optional[str] = None,
            verification_mode: str = "small_llm",
            verification_llm_provider: str = "ollama",
            verification_model_name: str = "llama3.2:3b",
            verification_server: Optional[str] = None,
            provided_evidence: Optional[Dict[str, str]] = None,
            reasoning_steps: int = 3,
            confidence_threshold: float = 0.7,
    ):
        self.decomposer = initialize_decomposer(
            decomposition_mode,
            decomposition_llm_provider,
            decomposition_model_name,
            decomposition_server,
            reasoning_steps,
        )

        self.verifier = initialize_verifier(
            verification_mode,
            verification_llm_provider,
            verification_model_name,
            verification_server,
            provided_evidence,
            confidence_threshold,
            reasoning_steps,
        )

    def decompose(
            self,
            dataset: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enhanced decomposition with reasoning for small LLMs"""
        # Split each response
        decomposer_input = []
        for item in dataset:
            sentences = parse_sentences(item['response'])
            for idx, sentence in enumerate(sentences):
                decomposer_input.append({
                    "id": item["id"],
                    "sentence_id": idx,
                    "context": item['response'],
                    "sentence": sentence['text'].strip(),
                })

        decompositions = self.decomposer.do_decompose(decomposer_input)
        return decompositions

    def verify(self, decompositions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced verification with reasoning for small LLMs"""
        non_empty_decompositions = [d for d in decompositions if d["claim"] is not None]
        verifier_output = self.verifier.do_verify(non_empty_decompositions)
        return verifier_output


def parse_args():
    parser = ArgumentParser(description="MedScore Implementation with Multiple Modes Support")

    # Small LLM: gemma3:27b, gpt-oss:20b, gemma3:12b, llama3.1:8b, llama3.2:3b

    # General
    parser.add_argument("--input_file", required=True, type=str, help="Input JSONL file with medical responses")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for results")
    parser.add_argument("--decompose_only", action="store_true", help="Only run decomposition")
    parser.add_argument("--verify_only", action="store_true", help="Only run verification")

    # Decomposition
    parser.add_argument("--decomposition_mode", type=str,
                        choices=["small_llm", "medscore", "factscore", "dndscore", "custom"],
                        default="small_llm", help="Decomposition mode")
    parser.add_argument("--decomposition_llm_provider", type=str, choices=["ollama", "openapi"],
                        default="ollama", help="LLM provider for decomposition")
    parser.add_argument("--decomposition_model_name", type=str, default="gemma3:12b",
                        help="Model name for decomposition")
    parser.add_argument("--decomposition_server", type=str, default=None,
                        help="Server URL for decomposition LLM")
    parser.add_argument("--decomposition_input_file", type=str, default=None,
                        help="Path to decomposition input file (required for verify_only mode)")
    parser.add_argument("--decomposition_prompt_path", type=str, default=None,
                        help="Path to custom decomposition prompt")

    # Verification
    parser.add_argument("--verification_mode", type=str,
                        choices=["internal", "provided", "internal_small_llm", "provided_small_llm"],
                        default="internal_small_llm", help="Verification mode")
    parser.add_argument("--verification_llm_provider", type=str, choices=["ollama", "openapi"],
                        default="ollama", help="LLM provider for verification")
    parser.add_argument("--verification_model_name", type=str, default="gemma3:12b",
                        help="Model name for verification")
    parser.add_argument("--verification_server", type=str, default=None,
                        help="Server URL for verification LLM")
    parser.add_argument("--provided_evidence_path", type=str, default=None,
                        help="Path to provided evidence file (required for 'provided' mode)")

    # Small LLM specific parameters
    parser.add_argument("--reasoning_steps", type=int, default=3,
                        help="Number of reasoning steps for small LLMs")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for verification")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with jsonlines.open(args.input_file) as reader:
        dataset = [item for item in reader.iter()]

    # Handle provided evidence for 'provided' verification mode
    provided_evidence = None
    if args.verification_mode == "provided" or args.verification_mode == "provided_small_llm":
        if args.provided_evidence_path is not None:
            with open(args.provided_evidence_path, "r") as f:
                provided_evidence = json.load(f)
        else:
            raise InvalidArgumentException("Provided evidence path is required when verification_mode is 'provided'")

    # Set output file names based on modes
    mode_prefix = f"{args.decomposition_mode}_{args.verification_mode}"
    decomposition_output_file = os.path.join(args.output_dir, f"{mode_prefix}_decompositions.jsonl")
    verification_output_file = os.path.join(args.output_dir, f"{mode_prefix}_verifications.jsonl")
    output_file = os.path.join(args.output_dir, f"{mode_prefix}_med_score_output.jsonl")

    scorer = MedScoreSmallLLM(
        decomposition_mode=args.decomposition_mode,
        decomposition_llm_provider=args.decomposition_llm_provider,
        decomposition_model_name=args.decomposition_model_name,
        decomposition_server=args.decomposition_server,
        decomposition_prompt_path=args.decomposition_prompt_path,
        verification_mode=args.verification_mode,
        verification_llm_provider=args.verification_llm_provider,
        verification_model_name=args.verification_model_name,
        verification_server=args.verification_server,
        provided_evidence=provided_evidence,
        reasoning_steps=args.reasoning_steps,
        confidence_threshold=args.confidence_threshold,
    )

    # Process decomposition
    decompositions = []
    if not args.verify_only:
        print(f"Running decomposition with {args.decomposition_mode} mode...")
        decompositions = scorer.decompose(dataset)
        with jsonlines.open(decomposition_output_file, 'w') as writer:
            writer.write_all(decompositions)

        if args.decompose_only:
            print(f"Decomposition completed. Results saved to {decomposition_output_file}")
            exit(0)
    else:
        # Load existing decompositions
        with jsonlines.open(args.decomposition_input_file, 'r') as reader:
            decompositions = [item for item in reader.iter()]

    # Process verification
    print(f"Running verification with {args.verification_mode} mode...")
    verifications = scorer.verify(decompositions)
    with jsonlines.open(verification_output_file, 'w') as writer:
        writer.write_all(verifications)

    # Combine results
    combined_output = {
        d["id"]: {
            "id": d["id"],
            "claims": []
        } for d in decompositions
    }
    for verification in verifications:
        claim_info = {
            k: v for k, v in verification.items() if k not in {"id", "sentence_id", "claim_id"}
        }
        combined_output[verification['id']]['claims'].append(claim_info)

    # Aggregate scores
    for idx in combined_output:
        claim_scores = [claim['score'] for claim in combined_output[idx]['claims']]
        if len(claim_scores) == 0:
            combined_output[idx]["score"] = None
        else:
            combined_output[idx]["score"] = sum(claim_scores) / len(claim_scores)

    combined_output = [v for k, v in combined_output.items()]
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(combined_output)

    # Calculate and display final metrics
    scores = [item['score'] for item in combined_output if item['score'] is not None]
    final_score = sum(scores) / len(scores) if scores else None

    print(f"\n=== MedScore Results ({args.decomposition_mode} + {args.verification_mode}) ===")
    print(f"Total responses evaluated: {len(combined_output)}")
    print(f"Responses with valid scores: {len(scores)}")
    print(f"Final MedScore: {final_score:.4f}")
    print(f"Results saved to: {output_file}")

    # Additional metrics for comparison
    if scores:
        import statistics

        print(f"Score statistics:")
        print(f"  Mean: {statistics.mean(scores):.4f}")
        print(f"  Median: {statistics.median(scores):.4f}")
        print(f"  Std Dev: {statistics.stdev(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
