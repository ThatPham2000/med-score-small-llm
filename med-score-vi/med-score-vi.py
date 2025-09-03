import json
import os
from argparse import ArgumentParser

import jsonlines

from exceptions import InvalidArgumentException


class MedScoreVi(object):
    def __init__(
            self,
            model_name_decomposition: str,
            server_decomposition: str,
            model_name_verification: str,
            server_verification: str,
            verification_mode: str,
            decomposition_mode: str,
            response_key: str,
            decomposition_prompt_path: str = None,
            provided_evidence_path: str = None
    ):
        self.response_key = response_key
        self.decomposition_prompt_path = decomposition_prompt_path
        self.provided_evidence_path = provided_evidence_path
        # Initialize decomposer and verifier here
        # self.decomposer = ...
        # self.verifier = ...


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
    parser.add_argument("--decomposition_server", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--decomposition_prompt_path", type=str, default=None)

    # Verification
    parser.add_argument("--verification_mode", type=str, choices=["internal", "provided"], default="internal")
    parser.add_argument("--verification_llm_provider", type=str, choices=["ollama", "openapi"], default="ollama")
    parser.add_argument("--verification_model_name", type=str, default="gpt-oss:20b")
    parser.add_argument("--verification_server", type=str, default="https://api.openai.com/v1")
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

    decomposition_output_file = os.path.join(args.output_dir, "decompositions.jsonl")
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

    if args.decomposition_mode == "custom" and not args.decomp_prompt_path:
        raise InvalidArgumentException("Must provide a decomposition prompt path with CustomDecomposer")

    # Initialize MedScore
    scorer = MedScoreVi(
        model_name_decomposition=args.decomposition_model_name,
        server_decomposition=args.server_decomposition,
        model_name_verification=args.model_name_verification,
        server_verification=args.server_verification,
        verification_mode=args.verification_mode,
        decomposition_mode=args.decomposition_mode,
        response_key=args.response_key,
        provided_evidence=provided_evidence,
        custom_decomposition_prompt_path=args.decomp_prompt_path
    )

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
    verifications = scorer.verify(decompositions)
    with jsonlines.open(verification_output_file, 'w') as writer:
        writer.write_all(verifications)
