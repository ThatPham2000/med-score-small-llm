import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

from jsonlines import jsonlines

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unified_pipeline.llm_provider import create_llm_provider, LLMProvider
from unified_pipeline.unified_pipeline import create_pipeline


# df = pd.read_parquet("hf://datasets/valen02/logiqa-en_AGIEval/data/test-00000-of-00001.parquet")
def read_agieval_logiqa_rows() -> List[Dict[str, str]]:
    with jsonlines.open(
            '/Users/that.phamvan/my_ws/master/med-score-small-llm/agieval-logiqa-en/agieval-logiqa-en.jsonl',
            'r') as reader:
        rows = [item for item in reader.iter()]
    return rows


def build_logiqa_prompt(query: str, choices: List[str]) -> str:
    """Build prompt for LogiQA reasoning tasks."""
    choices_block = "\n".join([f"- {choice}" for choice in choices])
    # return (
    #     "You are a logical reasoning assistant. Analyze the given problem and select the SINGLE best answer choice.\n"
    #     "Instructions:\n"
    #     "- Think step by step through the logical reasoning\n"
    #     "- Consider all given information carefully\n"
    #     "- Select the most logical answer from the provided choices\n"
    #     "- Respond with ONLY the chosen option text, no extra words\n\n"
    #     f"Problem: {query}\n\n"
    #     f"Answer Choices:\n{choices_block}\n\n"
    #     "Respond with EXACTLY the chosen option text."
    # )
    return f"""Query: {query}
    
Answer Choices:\n{choices_block}

Respond with EXACTLY the chosen option text.
    """


def extract_choice_text(response: str, choices: List[str]) -> str:
    text = (response or "").strip()
    if not text:
        return ""

    # Exact match first
    for choice in choices:
        if text == choice:
            return choice

    # Check for letter-only responses (A, B, C, D, etc.)
    if len(text) == 1 and text.isalpha():
        letter = text.upper()
        for choice in choices:
            if choice.startswith(f"({letter})") or choice.startswith(f"{letter} "):
                return choice

    # Check for letter with parenthesis responses ((A), (B), etc.)
    if text.startswith("(") and text.endswith(")"):
        letter = text[1:-1].upper()
        if letter.isalpha():
            for choice in choices:
                if choice.startswith(f"({letter})") or choice.startswith(f"{letter} "):
                    return choice

    # Check for letter with closing parenthesis A), B), etc.)
    if text.endswith(")") and len(text) == 2:
        letter = text[:-1].upper()
        if letter.isalpha():
            for choice in choices:
                if choice.startswith(f"({letter})") or choice.startswith(f"{letter} "):
                    return choice

    # Check if response contains the text part of any choice (without letter prefix)
    for choice in choices:
        # Extract text part after the letter prefix like "(A)", "(B)", etc.
        if choice.startswith("(") and ")" in choice:
            choice_text = choice.split(")", 1)[1].strip()
            if choice_text and choice_text in text:
                return choice

    # Fuzzy contains (fall back)
    lower = text.lower()
    best = ""
    for choice in choices:
        if choice.lower() in lower:
            best = choice
            break
    return best


def run_pipeline_eval(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    """Run evaluation using the unified pipeline."""
    pipeline = create_pipeline(
        llm,
        enable_atomic_fact_decomposition=False,
        verbose=True,
        temperature=0.1,
        reasoning_strategy="cot",
        use_rag=False,
        enable_tools=False,
        auto_config=False
    )

    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            query = row.get("query", "")
            choices = row.get("choices", [])
            gold_idx = row.get("gold", [0])[0] if row.get("gold") else 0
            correct_answer = choices[gold_idx] if gold_idx < len(choices) else ""

            prompt = build_logiqa_prompt(query, choices)
            start = time.time()
            result = pipeline.process(prompt)
            elapsed = time.time() - start

            model_answer = result.get("final_answer", "")
            picked = extract_choice_text(model_answer, choices)
            is_correct = picked == correct_answer
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "query": query,
                "choices": choices,
                "correct_answer": correct_answer,
                "correct_answer_index": gold_idx,
                "pipeline_final_answer": model_answer,
                "picked": picked,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def run_llm_only_eval(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    """Run evaluation using LLM only (no pipeline)."""
    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            query = row.get("query", "")
            choices = row.get("choices", [])
            gold_idx = row.get("gold", [0])[0] if row.get("gold") else 0
            correct_answer = choices[gold_idx] if gold_idx < len(choices) else ""

            prompt = build_logiqa_prompt(query, choices)
            start = time.time()
            response = llm.generate(prompt, temperature=0.1, max_tokens=1024)
            elapsed = time.time() - start

            picked = extract_choice_text(response, choices)
            is_correct = picked == correct_answer
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "query": query,
                "choices": choices,
                "correct_answer": correct_answer,
                "correct_answer_index": gold_idx,
                "llm_response": response,
                "picked": picked,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate AGIEval LogiQA dataset with unified pipeline and LLM-only.")
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model to use for both evaluations",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "output_csv_test")),
        help="Directory to write outputs",
    )
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rows = read_agieval_logiqa_rows()
    print(f"Loaded {len(rows)} rows")

    # No shuffling needed

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
        "dataset": 'hf://datasets/valen02/logiqa-en_AGIEval/data/test-00000-of-00001.parquet',
        "model": args.model,
        "pipeline": {"correct": p_correct, "total": p_total, "accuracy": round(p_acc, 4)},
        "llm_only": {"correct": l_correct, "total": l_total, "accuracy": round(l_acc, 4)},
        "comparison": {
            "pipeline_minus_llm_only": round(p_acc - l_acc, 4)
        }
    }
    with open(os.path.join(outdir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote summary:", os.path.join(outdir, "comparison.json"))


if __name__ == "__main__":
    main()
