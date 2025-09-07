import json
import os
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import jsonlines
import ollama

from decomposer_fact_score import DecomposerFactScore
from decomposer_med_score import DecomposerMedScore
from exceptions import InvalidArgumentException, IllegalArgumentException
from llm_ollama import LLMOllama
from utils import parse_sentences


def initialize_llm(llm_provider: str, model_name: str, server: Optional[str]):
    llm_provider = llm_provider.lower()
    if llm_provider == "ollama":
        if server is None:
            return LLMOllama(model_name=model_name)
        return LLMOllama(
            model_name=model_name,
            ollama_client=ollama.AsyncClient(host=server, verify=False)
        )

    if llm_provider == "openapi":
        pass
    raise IllegalArgumentException(f"Unknown LLM provider: {llm_provider}")


def initialize_decomposition(
        decomposition_mode: str,
        decomposition_llm_provider: str,
        decomposition_model_name: str,
        decomposition_server: str,
):
    mode = decomposition_mode.lower()
    llm = initialize_llm(decomposition_llm_provider, decomposition_model_name, decomposition_server)
    if mode == "medscore":
        return DecomposerMedScore(llm)
    if mode == "factscore":
        return DecomposerFactScore(llm)
    raise IllegalArgumentException(f"Unknown decomposition mode: {mode}")


class MedScoreVi(object):
    def __init__(
            self,
            decomposition_mode: str,
            decomposition_llm_provider: str,
            decomposition_model_name: str,
            decomposition_server: str,
            decomposition_prompt_path: None,
            verification_mode: str,
            verification_llm_provider: str,
            verification_model_name: str,
            verification_server: str,
            provided_evidence_path: None,
    ):
        self.decomposer = initialize_decomposition(
            decomposition_mode,
            decomposition_llm_provider,
            decomposition_model_name,
            decomposition_server,
        )

    def decompose(
            self,
            dataset: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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


def parse_args():
    parser = ArgumentParser()

    # General
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--decompose_only", action="store_true")
    parser.add_argument("--verify_only", action="store_true")

    # Decomposition
    parser.add_argument("--decomposition_mode", type=str, choices=["medscore", "factscore", "dndscore", "custom"],
                        default="medscore")
    parser.add_argument("--decomposition_llm_provider", type=str, choices=["ollama", "openapi"], default="ollama")
    parser.add_argument("--decomposition_model_name", type=str, default="gpt-oss:20b")
    parser.add_argument("--decomposition_server", type=str, default=None)
    parser.add_argument("--decomposition_prompt_path", type=str, default=None)

    # Verification
    parser.add_argument("--verification_mode", type=str, choices=["internal", "provided"], default="internal")
    parser.add_argument("--verification_llm_provider", type=str, choices=["ollama", "openapi"], default="ollama")
    parser.add_argument("--verification_model_name", type=str, default="gpt-oss:20b")
    parser.add_argument("--verification_server", type=str, default=None)
    parser.add_argument("--provided_evidence_path", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with jsonlines.open(args.input_file) as reader:
        dataset = [item for item in reader.iter()]
    # dataset = [dataset[0]]

    decomposition_output_file = os.path.join(args.output_dir, "decompositions_dmm.jsonl")
    verification_output_file = os.path.join(args.output_dir, "verifications.jsonl")
    output_file = os.path.join(args.output_dir, "med_score_vi_output.jsonl")

    if args.verification_mode == "provided":
        if args.provided_evidence_path is not None:
            with open(args.provided_evidence_path, "r") as f:
                provided_evidence = json.load(f)
        else:
            raise Exception("Provided evidence path is required when verification_mode is 'provided'")
    else:
        provided_evidence = None

    if args.decomposition_mode == "custom" and not args.decomposition_prompt_path:
        raise InvalidArgumentException("Must provide a decomposition prompt path with CustomDecomposer")

    scorer = MedScoreVi(
        decomposition_mode=args.decomposition_mode,
        decomposition_llm_provider=args.decomposition_llm_provider,
        decomposition_model_name=args.decomposition_model_name,
        decomposition_server=args.decomposition_server,
        decomposition_prompt_path=args.decomposition_prompt_path,
        verification_mode=args.verification_mode,
        verification_llm_provider=args.verification_llm_provider,
        verification_model_name=args.verification_model_name,
        verification_server=args.verification_server,
        provided_evidence_path=args.provided_evidence_path)

    # Process decomposition
    if not args.verify_only:
        decompositions = scorer.decompose(dataset)
        with jsonlines.open(decomposition_output_file, 'w') as writer:
            writer.write_all(decompositions)

        if args.decompose_only:
            exit(0)
    else:
        with jsonlines.open(os.path.join(args.output_dir, "decompositions.jsonl"), 'r') as reader:
            decompositions = [item for item in reader.iter()]

    # Process verification
    # verifications = scorer.verify(decompositions)
    # with jsonlines.open(verification_output_file, 'w') as writer:
    #     writer.write_all(verifications)
