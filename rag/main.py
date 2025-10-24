import argparse
import json
import os
import sys
import time
import uuid
from typing import List, Dict, Tuple

import pandas as pd
from jsonlines import jsonlines

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unified_pipeline.llm_provider import create_llm_provider, LLMProvider
from unified_pipeline.unified_pipeline import create_pipeline


def download_rag_dataset():
    splits = {'train': 'data/train-00000-of-00001-f0c158413defd454.parquet',
              'test': 'data/test-00000-of-00001-06d83c58a8ea10e8.parquet'}
    df = pd.read_parquet("hf://datasets/neural-bridge/rag-dataset-1200/" + splits["test"])

    # Add UUID column as first column
    df.insert(0, 'uuid', [str(uuid.uuid4()) for _ in range(len(df))])

    print(df.head())
    print(f"Total rows: {len(df)}")

    # save to jsonl
    df.to_json("rag_test.jsonl", orient="records", lines=True)
    print("Saved to rag_test.jsonl")


def read_rag_rows() -> List[Dict[str, str]]:
    with jsonlines.open(
            '/Users/that.phamvan/my_ws/master/med-score-small-llm/rag/rag_test.jsonl',
            'r') as reader:
        rows = [item for item in reader.iter()]
    return rows[100:150]


def check_correct_answer(context: str, question: str, correct_answer: str, model_answer: str, llm: LLMProvider) -> bool:
    """Use LLM to check if the model answer is correct based on context and correct answer."""
    validation_prompt = f"""You are an expert evaluator. Your task is to determine if the model's answer is correct based on the provided context and the correct answer.

Context: {context}

Question: {question}

Correct Answer: {correct_answer}

Model's Answer: {model_answer}

Instructions:
1. Compare the model's answer with the correct answer
2. Consider if the model's answer is factually correct based on the context
3. Allow for different phrasings or wordings that convey the same meaning
4. Consider if the model's answer captures the essential information from the correct answer
5. Be strict about factual accuracy but flexible about wording

Was the model's answer correct?

Respond with ONLY "CORRECT" or "INCORRECT" - no other text."""

    try:
        response = llm.generate(validation_prompt, temperature=0.1, max_tokens=50)
        response = response.strip().upper()
        return response == "CORRECT"
    except Exception as e:
        print(f"Error in check_correct_answer: {e}")
        # Fallback to simple string comparison
        return model_answer.lower().strip() == correct_answer.lower().strip()


def run_pipeline_eval_with_rag(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    """Run evaluation using the unified pipeline with RAG."""
    pipeline = create_pipeline(
        llm,
        enable_atomic_fact_decomposition=False,
        verbose=True,
        temperature=0.1,
        reasoning_strategy="cot",
        use_rag=True,  # Enable RAG for pipeline
        documents_path='/Users/that.phamvan/my_ws/master/med-score-small-llm/rag/rag_test.jsonl',
        enable_tools=False,
        auto_config=False
    )

    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            context = row.get("context", "")
            question = row.get("question", "")
            correct_answer = row.get("answer", "")

            prompt = f"""Question: {question}

Based on the provided context, please answer the question accurately and concisely."""
            start = time.time()
            result = pipeline.process(prompt)
            elapsed = time.time() - start

            model_answer = result.get("final_answer", "").strip()
            is_correct = check_correct_answer(context, question, correct_answer, model_answer, llm)
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "uuid": row.get("uuid", ""),
                "context": context,
                "question": question,
                "correct_answer": correct_answer,
                "reasoning_trace": result.get("reasoning_trace", []),
                "pipeline_final_answer": model_answer,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def run_llm_only_eval(rows: List[Dict[str, str]], llm: LLMProvider, output_path: str) -> Tuple[int, int]:
    """Run evaluation using LLM only (no RAG, no pipeline)."""
    correct = 0
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            context = row.get("context", "")
            question = row.get("question", "")
            correct_answer = row.get("answer", "")

            # Simple prompt without context for LLM-only evaluation
            prompt = f"""Question: {question}

Please answer the question accurately and concisely."""

            start = time.time()
            response = llm.generate(prompt, temperature=0.1, max_tokens=2000)
            elapsed = time.time() - start

            model_answer = response.strip()
            # Use LLM validation with context to check correctness
            is_correct = check_correct_answer(context, question, correct_answer, model_answer, llm)
            correct += 1 if is_correct else 0
            total += 1

            out = {
                "index": idx,
                "uuid": row.get("uuid", ""),
                "context": context,
                "question": question,
                "correct_answer": correct_answer,
                "llm_response": model_answer,
                "is_correct": is_correct,
                "latency_s": round(elapsed, 3),
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return correct, total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG dataset with unified pipeline (with RAG) and LLM-only (without RAG).")
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model to use for both evaluations",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "output_rag_comparison")),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the RAG dataset first",
    )
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Download dataset if requested
    if args.download:
        print("Downloading RAG dataset...")
        download_rag_dataset()

    # Load RAG dataset
    rows = read_rag_rows()
    print(f"Loaded {len(rows)} rows from RAG dataset")

    # Initialize LLM (Ollama - gemma3:12b by default)
    llm = create_llm_provider("ollama", model=args.model)

    # Run pipeline evaluation with RAG
    pipeline_out = os.path.join(outdir, "pipeline_with_rag_results.jsonl")
    p_correct, p_total = run_pipeline_eval_with_rag(rows, llm, pipeline_out)
    p_acc = p_correct / p_total if p_total else 0.0
    print(f"Pipeline with RAG accuracy: {p_correct}/{p_total} = {p_acc:.3f}")

    # Run LLM-only evaluation (without RAG)
    llm_out = os.path.join(outdir, "llm_only_results.jsonl")
    l_correct, l_total = run_llm_only_eval(rows, llm, llm_out)
    l_acc = l_correct / l_total if l_total else 0.0
    print(f"LLM-only accuracy: {l_correct}/{l_total} = {l_acc:.3f}")

    # Write summary
    summary = {
        "dataset": "neural-bridge/rag-dataset-1200",
        "model": args.model,
        "pipeline_with_rag": {"correct": p_correct, "total": p_total, "accuracy": round(p_acc, 4)},
        "llm_only": {"correct": l_correct, "total": l_total, "accuracy": round(l_acc, 4)},
        "comparison": {
            "pipeline_with_rag_minus_llm_only": round(p_acc - l_acc, 4),
            "rag_benefit": "Pipeline with RAG uses context field, LLM-only uses only question field"
        }
    }
    with open(os.path.join(outdir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote summary:", os.path.join(outdir, "comparison.json"))


if __name__ == '__main__':
    main()
