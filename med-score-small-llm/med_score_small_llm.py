import json
import os
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import jsonlines
import ollama

from decomposer_small_llm import DecomposerSmallLLM
from exceptions import InvalidArgumentException, IllegalArgumentException
from llm_ollama import LLMOllama
from llm_openai import LLMOpenAI
from utils import parse_sentences
from verifier_small_llm import VerifierSmallLLM


def initialize_llm(llm_provider: str, model_name: str, server: Optional[str]):
    """Initialize LLM with enhanced parameters for small models"""
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


def initialize_small_llm_decomposer(
        decomposition_llm_provider: str,
        decomposition_model_name: str,
        decomposition_server: Optional[str],
        reasoning_steps: int = 3,
        use_chain_of_thought: bool = True,
):
    """Initialize enhanced decomposer for small language models"""
    llm = initialize_llm(decomposition_llm_provider, decomposition_model_name, decomposition_server)
    return DecomposerSmallLLM(
        llm=llm,
        reasoning_steps=reasoning_steps,
        use_chain_of_thought=use_chain_of_thought
    )


def initialize_small_llm_verifier(
        verification_llm_provider: str,
        verification_model_name: str,
        verification_server: Optional[str],
        use_chain_of_thought: bool = True,
        confidence_threshold: float = 0.7,
        reasoning_steps: int = 3,
):
    """Initialize enhanced verifier for small language models"""
    llm = initialize_llm(verification_llm_provider, verification_model_name, verification_server)
    return VerifierSmallLLM(
        llm=llm,
        use_chain_of_thought=use_chain_of_thought,
        confidence_threshold=confidence_threshold,
        reasoning_steps=reasoning_steps
    )


class MedScoreSmallLLM(object):
    """
    Enhanced MedScore implementation for small language models.
    
    This class implements the approach "Small Language Models for Factuality Evaluation 
    of Free-Form Medical Answers" by making small language models more intelligent and 
    reasoning like large language models through:
    
    1. Chain-of-thought prompting
    2. Multi-step reasoning processes
    3. Enhanced decomposition with reasoning steps
    4. Improved verification with confidence scoring
    5. Context-aware fact extraction
    """
    
    def __init__(
            self,
            decomposition_llm_provider: str,
            decomposition_model_name: str,
            decomposition_server: Optional[str],
            verification_llm_provider: str,
            verification_model_name: str,
            verification_server: Optional[str],
            reasoning_steps: int = 3,
            use_chain_of_thought: bool = True,
            confidence_threshold: float = 0.7,
    ):
        self.decomposer = initialize_small_llm_decomposer(
            decomposition_llm_provider,
            decomposition_model_name,
            decomposition_server,
            reasoning_steps,
            use_chain_of_thought,
        )

        self.verifier = initialize_small_llm_verifier(
            verification_llm_provider,
            verification_model_name,
            verification_server,
            use_chain_of_thought,
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
    parser = ArgumentParser(description="MedScore Small LLM Implementation")

    # General
    parser.add_argument("--input_file", required=True, type=str, help="Input JSONL file with medical responses")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for results")
    parser.add_argument("--decompose_only", action="store_true", help="Only run decomposition")
    parser.add_argument("--verify_only", action="store_true", help="Only run verification")

    # Decomposition
    parser.add_argument("--decomposition_llm_provider", type=str, choices=["ollama", "openapi"], 
                        default="ollama", help="LLM provider for decomposition")
    parser.add_argument("--decomposition_model_name", type=str, default="llama3.2:3b", 
                        help="Model name for decomposition (small model)")
    parser.add_argument("--decomposition_server", type=str, default=None, 
                        help="Server URL for decomposition LLM")

    # Verification
    parser.add_argument("--verification_llm_provider", type=str, choices=["ollama", "openapi"], 
                        default="ollama", help="LLM provider for verification")
    parser.add_argument("--verification_model_name", type=str, default="llama3.2:3b", 
                        help="Model name for verification (small model)")
    parser.add_argument("--verification_server", type=str, default=None, 
                        help="Server URL for verification LLM")

    # Small LLM specific parameters
    parser.add_argument("--reasoning_steps", type=int, default=3, 
                        help="Number of reasoning steps for small LLMs")
    parser.add_argument("--use_chain_of_thought", action="store_true", default=True, 
                        help="Use chain-of-thought prompting")
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

    decomposition_output_file = os.path.join(args.output_dir, "small_llm_decompositions.jsonl")
    verification_output_file = os.path.join(args.output_dir, "small_llm_verifications_evidence.jsonl")
    output_file = os.path.join(args.output_dir, "small_llm_med_score_output.jsonl")

    scorer = MedScoreSmallLLM(
        decomposition_llm_provider=args.decomposition_llm_provider,
        decomposition_model_name=args.decomposition_model_name,
        decomposition_server=args.decomposition_server,
        verification_llm_provider=args.verification_llm_provider,
        verification_model_name=args.verification_model_name,
        verification_server=args.verification_server,
        reasoning_steps=args.reasoning_steps,
        use_chain_of_thought=args.use_chain_of_thought,
        confidence_threshold=args.confidence_threshold,
    )

    # Process decomposition
    decompositions = []
    if not args.verify_only:
        print("Running enhanced decomposition with small LLM...")
        decompositions = scorer.decompose(dataset)
        with jsonlines.open(decomposition_output_file, 'w') as writer:
            writer.write_all(decompositions)

        if args.decompose_only:
            print(f"Decomposition completed. Results saved to {decomposition_output_file}")
            exit(0)
    else:
        with jsonlines.open(os.path.join(args.output_dir, "small_llm_decompositions.jsonl"), 'r') as reader:
            decompositions = [item for item in reader.iter()]

    # Process verification
    print("Running enhanced verification with small LLM...")
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
    
    print(f"\n=== Small LLM MedScore Results ===")
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
