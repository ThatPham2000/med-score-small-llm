"""
Measure run-to-run variation of the repeated classification runs produced by repeat_classification.py.

Reports:
- raw-text determinism: share of claims whose raw response is byte-identical in every run
- label stability: per-claim majority share (p_max), label entropy, number of distinct labels
- Fleiss' kappa across runs (runs as raters)
- per-run Valid rate and agreement with human labels: mean, std, min, max
"""
import glob
import json
import math
import os
from argparse import ArgumentParser
from collections import Counter

import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def claim_key(c):
    return f'{c["id"]}|{c["sentence_id"]}|{c["claim_id"]}'


def fleiss_kappa(label_matrix):
    # label_matrix: n_items x n_raters
    categories = sorted({lab for row in label_matrix for lab in row})
    n_items, n_raters = len(label_matrix), len(label_matrix[0])
    counts = np.array([[Counter(row)[cat] for cat in categories] for row in label_matrix], dtype=float)
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    if p_e == 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def describe(values):
    values = np.array(values, dtype=float)
    return {"mean": round(values.mean(), 4), "std": round(values.std(ddof=1), 4) if len(values) > 1 else 0.0,
            "min": round(values.min(), 4), "max": round(values.max(), 4)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runs_dir", required=True, type=str)
    parser.add_argument("--human_file", type=str, default=None,
                        help="JSONL with manual_claim_quality_type for agreement with human labels")
    parser.add_argument("--original_file", type=str, default=None,
                        help="JSONL of the original thesis run, included as an extra run")
    args = parser.parse_args()

    run_files = sorted(glob.glob(os.path.join(args.runs_dir, "run_*.jsonl")))
    runs = [load_jsonl(p) for p in run_files]
    if args.original_file:
        runs = [load_jsonl(args.original_file)] + runs
    keys = [claim_key(c) for c in runs[0]]
    by_run = [{claim_key(c): c for c in run} for run in runs]
    n_runs = len(runs)

    labels = [[r[k]["claim_quality_type"] for r in by_run] for k in keys]
    raws = [[r[k]["raw_claim_quality_response"] for r in by_run] for k in keys]

    per_claim = []
    for k, labs, rs in zip(keys, labels, raws):
        counter = Counter(labs)
        p = np.array(list(counter.values())) / n_runs
        per_claim.append({
            "claim": k,
            "distribution": dict(counter),
            "majority": counter.most_common(1)[0][0],
            "p_max": round(p.max(), 4),
            "entropy_bits": round(float(-(p * np.log2(p)).sum()), 4),
            "n_distinct_labels": len(counter),
            "n_distinct_raw": len(set(rs)),
        })

    valid_rate = [np.mean([labs[i] == "Valid" for labs in labels]) for i in range(n_runs)]
    known_labels = {"Valid", "Unverifiable", "Incorrectly structured", "Context-dependent", "Hallucinated",
                    "Incomplete", "Redundant"}
    parse_failed = [
        {"run": os.path.basename(run_files[r - (n_runs - len(run_files))]) if r >= n_runs - len(run_files)
         else "original", "claim": keys[i], "label": labels[i][r]}
        for i in range(len(keys)) for r in range(n_runs) if labels[i][r] not in known_labels
    ]
    summary = {
        "n_runs": n_runs,
        "n_claims": len(keys),
        "parse_failed_count": len(parse_failed),
        "parse_failed": parse_failed,
        "raw_identical_all_runs": sum(c["n_distinct_raw"] == 1 for c in per_claim),
        "label_identical_all_runs": sum(c["n_distinct_labels"] == 1 for c in per_claim),
        "p_max": describe([c["p_max"] for c in per_claim]),
        "entropy_bits": describe([c["entropy_bits"] for c in per_claim]),
        "fleiss_kappa_across_runs": round(fleiss_kappa(labels), 4) if n_runs > 1 else None,
        "valid_rate_per_run": describe(valid_rate),
        "label_share_per_run": {
            lab: describe([np.mean([labs[i] == lab for labs in labels]) for i in range(n_runs)])
            for lab in sorted({l for labs in labels for l in labs})
        },
    }

    if args.human_file:
        human = {claim_key(c): c["manual_claim_quality_type"] for c in load_jsonl(args.human_file)}
        common = [i for i, k in enumerate(keys) if k in human]
        agreement = [np.mean([labels[i][r] == human[keys[i]] for i in common]) for r in range(n_runs)]
        majority_agreement = np.mean([per_claim[i]["majority"] == human[keys[i]] for i in common])
        summary["human_agreement"] = {
            "n_claims_with_human_label": len(common),
            "per_run": describe(agreement),
            "majority_vote": round(float(majority_agreement), 4),
        }

    with open(os.path.join(args.runs_dir, "per_claim_stability.json"), "w") as f:
        json.dump(per_claim, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.runs_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUnstable claims:")
    for c in per_claim:
        if c["n_distinct_labels"] > 1:
            print(f'  {c["claim"]}: {c["distribution"]} (p_max={c["p_max"]})')
