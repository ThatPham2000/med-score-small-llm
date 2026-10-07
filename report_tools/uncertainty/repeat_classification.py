"""
Repeat the taxonomy classification step R times with the exact experimental configuration
(seed=42, temperature=0.0, top_p=1.0, batch_size=8) to measure run-to-run variation.

Each run is saved to <output_dir>/run_XXX.jsonl, so the script can be stopped and resumed.
"""
import asyncio
import json
import os
import sys
import time
from argparse import ArgumentParser

import ollama
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "med-score-small-llm"))

from claim_quality_evaluation import ClaimQualityEvaluation  # noqa: E402
from llm_ollama import LLMOllama  # noqa: E402
from utils import chunker  # noqa: E402


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="gpt-oss:20b")
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def run_once(evaluator: ClaimQualityEvaluation, claims, batch_size: int):
    # Keep the stored other_claims so the prompt is byte-identical to the original run
    messages = [
        [{"role": "user", "content": evaluator.format_classification_input(
            c["context"], c["sentence"], c["claim"], c.get("other_claims") or [])}]
        for c in claims
    ]
    completions = []
    for batch in chunker(messages, batch_size):
        completions.extend(asyncio.run(evaluator.llm.batch_response(list(batch))))

    outputs = []
    for claim, raw in zip(claims, evaluator.llm.normalize_llm_response(completions)):
        raw = (raw or "").strip()
        outputs.append({
            "id": claim["id"],
            "sentence_id": claim["sentence_id"],
            "claim_id": claim["claim_id"],
            "raw_claim_quality_response": raw,
            "claim_quality_type": evaluator.parse_classify_claims_output(raw),
        })
    return outputs


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_file) as f:
        claims = [json.loads(line) for line in f if line.strip()]

    client = ollama.AsyncClient(host=args.server) if args.server else None
    llm = LLMOllama(model_name=args.model_name, ollama_async_client=client)
    # Separate instance: ClaimQualityEvaluation sets normalized_llm.max_tokens=10240,
    # which would otherwise override the classifier's max_tokens=32000
    normalized_llm = LLMOllama(model_name=args.model_name, ollama_async_client=client)
    evaluator = ClaimQualityEvaluation(llm=llm, normalized_llm=normalized_llm, is_only_classification=True,
                                       batch_size=args.batch_size)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"model_name": llm.model_name, "seed": llm.seed, "temperature": llm.temperature,
                   "top_p": llm.top_p, "max_tokens": llm.max_tokens, "batch_size": args.batch_size,
                   "input_file": args.input_file, "n_claims": len(claims)}, f, indent=2)

    for run_idx in tqdm(range(args.runs), desc="Runs"):
        run_file = os.path.join(args.output_dir, f"run_{run_idx:03d}.jsonl")
        if os.path.exists(run_file):
            continue
        start = time.time()
        outputs = run_once(evaluator, claims, args.batch_size)
        elapsed = time.time() - start
        tmp_file = run_file + ".tmp"
        with open(tmp_file, "w") as f:
            for o in outputs:
                o["run_seconds"] = round(elapsed, 2)
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        os.replace(tmp_file, run_file)
