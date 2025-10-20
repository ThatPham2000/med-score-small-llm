import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

import pandas as pd

from unified_pipeline.llm_provider import create_llm_provider, LLMProvider
from unified_pipeline.unified_pipeline import create_pipeline


def read_crass_rows(csv_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path, sep=';', engine='python')
    # Ensure expected columns exist; fill missing with empty strings
    for col in ["Premise", "QCC", "CorrectAnswer", "Answer1", "Answer2", "PossibleAnswer3"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    rs = df.to_dict(orient="records")

    return rs


def build_choice_prompt(premise: str, qcc: str, choices: List[str]) -> str:
    choices_block = "\n".join([f"- {i + 1}. {c}" for i, c in enumerate(choices)])
    return (
        "You are a medical reasoning assistant. Select the SINGLE best answer choice strictly from the provided options.\n"
        "Instructions:\n"
        "- Think briefly but respond with ONLY the chosen option text, no extra words.\n"
        "- Do not invent new options.\n\n"
        f"Premise: {premise}\n"
        f"Question/Claim/Context: {qcc}\n\n"
        f"Choices:\n{choices_block}\n\n"
        "Respond with EXACTLY the chosen option text."
    )


def extract_choice_text(response: str, choices: List[str]) -> str:
    text = (response or "").strip()
    if not text:
        return ""
    # Exact match first
    for c in choices:
        if text == c:
            return c

    # Fuzzy contains (fall back)
    lower = text.lower()
    best = ""
    for c in choices:
        if c.lower() in lower:
            best = c
            break
    return best


def run_pipeline_eval(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    pipeline = create_pipeline(llm, enable_atomic_fact_decomposition=False, verbose=True, temperature=0.1)

    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            premise = row.get("Premise", "")
            qcc = row.get("QCC", "")
            correct_answer = row.get("CorrectAnswer", "")
            a1 = row.get("Answer1", "")
            a2 = row.get("Answer2", "")
            pa3 = row.get("PossibleAnswer3", "")
            choices = [correct_answer, a1, a2]
            if isinstance(pa3, str) and pa3.strip():
                choices.append(pa3)
            shuffled = choices.copy()
            random.shuffle(shuffled)

            prompt = build_choice_prompt(premise, qcc, shuffled)
            start = time.time()
            result = pipeline.process(prompt)
            elapsed = time.time() - start
            model_answer = result.get("final_answer", "")
            picked = extract_choice_text(model_answer, shuffled)
            is_correct = picked == correct_answer
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "premise": premise,
                "qcc": qcc,
                "choices": shuffled,
                "correct_answer": correct_answer,
                "pipeline_final_answer": model_answer,
                "picked": picked,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def run_llm_only_eval(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            premise = row.get("Premise", "")
            qcc = row.get("QCC", "")
            correct_answer = row.get("CorrectAnswer", "")
            a1 = row.get("Answer1", "")
            a2 = row.get("Answer2", "")
            pa3 = row.get("PossibleAnswer3", "")
            choices = [correct_answer, a1, a2]
            if isinstance(pa3, str) and pa3.strip():
                choices.append(pa3)
            shuffled = choices.copy()
            random.shuffle(shuffled)

            prompt = build_choice_prompt(premise, qcc, shuffled)
            start = time.time()
            response = llm.generate(prompt, temperature=0.1, max_tokens=1024)
            elapsed = time.time() - start
            picked = extract_choice_text(response, shuffled)
            is_correct = picked == correct_answer
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "premise": premise,
                "qcc": qcc,
                "choices": shuffled,
                "correct_answer": correct_answer,
                "llm_response": response,
                "picked": picked,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRASS dataset with unified pipeline and LLM-only.")
    parser.add_argument(
        "--csv",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crass", "CRASS_FTM_main_data_set.csv")),
        help="Path to CRASS CSV dataset",
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model to use for both evaluations",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crass", "output_crass")),
        help="Directory to write outputs",
    )
    args = parser.parse_args()

    csv_path = args.csv
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rows = read_crass_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    # Initialize LLM (Ollama - gemma3:12b by default)
    llm = create_llm_provider("ollama", model=args.model)

    # Run pipeline evaluation
    pipeline_out = os.path.join(outdir, "pipeline_results.jsonl")
    p_correct, p_total = run_pipeline_eval(rows, llm, pipeline_out)
    p_acc = p_correct / p_total if p_total else 0.0
    print(f"Pipeline accuracy: {p_correct}/{p_total} = {p_acc:.3f}")

    # Run LLM-only evaluation
    llm_out = os.path.join(outdir, "llm_only_results.jsonl")
    l_correct, l_total = run_llm_only_eval(rows, llm, llm_out)
    l_acc = l_correct / l_total if l_total else 0.0
    print(f"LLM-only accuracy: {l_correct}/{l_total} = {l_acc:.3f}")

    # Write summary
    summary = {
        "dataset": csv_path,
        "model": args.model,
        "pipeline": {"correct": p_correct, "total": p_total, "accuracy": round(p_acc, 4)},
        "llm_only": {"correct": l_correct, "total": l_total, "accuracy": round(l_acc, 4)},
        "comparison": {
            "pipeline_minus_llm_only": round(p_acc - l_acc, 4)
        }
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote summary:", os.path.join(outdir, "summary.json"))


if __name__ == "__main__":
    main()
