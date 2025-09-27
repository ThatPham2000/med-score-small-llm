#!/usr/bin/env python3
"""
Script to run Small LLM MedScore evaluation and generate comprehensive results.

This script implements the approach "Small Language Models for Factuality Evaluation 
of Free-Form Medical Answers" by:
1. Running enhanced decomposition with chain-of-thought reasoning
2. Running enhanced verification with confidence scoring
3. Generating comprehensive metrics and visualizations
4. Comparing results with baseline MedScore
"""

import argparse
import os
import sys
from pathlib import Path

# Add the med-score-vi module to a path
current_dir = Path(__file__).parent.absolute()
med_score_vi_path = current_dir / "med-score-vi"

# Try multiple path resolution strategies
possible_paths = [
    med_score_vi_path,
    current_dir / "med-score-vi",
    Path.cwd() / "med-score-vi",
    Path.cwd() / "med-score-vi" / "med-score-vi"
]

for path in possible_paths:
    if path.exists() and (path / "med_score_small_llm.py").exists():
        med_score_vi_path = path
        break
else:
    raise ImportError(f"med-score-vi directory not found. Tried: {[str(p) for p in possible_paths]}")

sys.path.insert(0, str(med_score_vi_path))

try:
    from med_score_small_llm import MedScoreSmallLLM, parse_args as parse_small_llm_args
    from evaluation_metrics import compare_medscore_results
    from visualization import create_medscore_visualizations
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Looking for modules in: {med_score_vi_path}")
    print(f"Available files in {med_score_vi_path}: {list(med_score_vi_path.glob('*.py'))}")
    raise


def run_small_llm_evaluation(
        input_file: str,
        output_dir: str,
        decomposition_model: str = "llama3.2:3b",
        verification_model: str = "llama3.2:3b",
        baseline_file: str = None,
        create_visualizations: bool = True,
        reasoning_steps: int = 3,
        use_chain_of_thought: bool = True,
        confidence_threshold: float = 0.7
):
    """
    Run comprehensive small LLM MedScore evaluation.
    
    Args:
        input_file: Path to input JSONL file with medical responses
        output_dir: Directory to save results
        decomposition_model: Model name for decomposition
        verification_model: Model name for verification
        baseline_file: Path to baseline results for comparison
        create_visualizations: Whether to create visualizations
        reasoning_steps: Number of reasoning steps for small LLMs
        use_chain_of_thought: Whether to use chain-of-thought prompting
        confidence_threshold: Confidence threshold for verification
    """

    print("=" * 60)
    print("Small Language Models for Factuality Evaluation")
    print("of Free-Form Medical Answers")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize small LLM MedScore
    print(f"\nInitializing Small LLM MedScore with:")
    print(f"  Decomposition Model: {decomposition_model}")
    print(f"  Verification Model: {verification_model}")
    print(f"  Reasoning Steps: {reasoning_steps}")
    print(f"  Chain-of-Thought: {use_chain_of_thought}")
    print(f"  Confidence Threshold: {confidence_threshold}")

    scorer = MedScoreSmallLLM(
        decomposition_llm_provider="ollama",
        decomposition_model_name=decomposition_model,
        decomposition_server=None,
        verification_llm_provider="ollama",
        verification_model_name=verification_model,
        verification_server=None,
        reasoning_steps=reasoning_steps,
        use_chain_of_thought=use_chain_of_thought,
        confidence_threshold=confidence_threshold,
    )

    # Load input data
    print(f"\nLoading input data from: {input_file}")
    import jsonlines
    with jsonlines.open(input_file) as reader:
        dataset = [item for item in reader.iter()]
    print(f"Loaded {len(dataset)} medical responses")

    # Run decomposition
    print("\n" + "=" * 40)
    print("Running Enhanced Decomposition")
    print("=" * 40)
    decompositions = scorer.decompose(dataset)

    # Save decompositions
    decomposition_file = os.path.join(output_dir, "small_llm_decompositions.jsonl")
    with jsonlines.open(decomposition_file, 'w') as writer:
        writer.write_all(decompositions)
    print(f"Decompositions saved to: {decomposition_file}")

    # Run verification
    print("\n" + "=" * 40)
    print("Running Enhanced Verification")
    print("=" * 40)
    verifications = scorer.verify(decompositions)

    # Save verifications
    verification_file = os.path.join(output_dir, "small_llm_verifications.jsonl")
    with jsonlines.open(verification_file, 'w') as writer:
        writer.write_all(verifications)
    print(f"Verifications saved to: {verification_file}")

    # Combine results
    print("\n" + "=" * 40)
    print("Combining Results")
    print("=" * 40)

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

    # Save final results
    output_file = os.path.join(output_dir, "small_llm_med_score_output.jsonl")
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(combined_output)
    print(f"Final results saved to: {output_file}")

    # Calculate and display metrics
    print("\n" + "=" * 40)
    print("Results Summary")
    print("=" * 40)

    scores = [item['score'] for item in combined_output if item['score'] is not None]
    final_score = sum(scores) / len(scores) if scores else None

    print(f"Total responses evaluated: {len(combined_output)}")
    print(f"Responses with valid scores: {len(scores)}")
    print(f"Final MedScore: {final_score:.4f}")

    if scores:
        import statistics
        print(f"\nScore Statistics:")
        print(f"  Mean: {statistics.mean(scores):.4f}")
        print(f"  Median: {statistics.median(scores):.4f}")
        print(f"  Std Dev: {statistics.stdev(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")

    # Compare with baseline if provided
    if baseline_file and os.path.exists(baseline_file):
        print("\n" + "=" * 40)
        print("Comparing with Baseline")
        print("=" * 40)

        comparison_file = os.path.join(output_dir, "comparison_report.json")
        compare_medscore_results(baseline_file, output_file, comparison_file)

    # Create visualizations
    if create_visualizations and baseline_file and os.path.exists(baseline_file):
        print("\n" + "=" * 40)
        print("Creating Visualizations")
        print("=" * 40)

        viz_dir = os.path.join(output_dir, "visualizations")
        create_medscore_visualizations(baseline_file, output_file, viz_dir)

    print(f"\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return {
        'final_score': final_score,
        'total_responses': len(combined_output),
        'valid_scores': len(scores),
        'output_file': output_file
    }


def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(
        description="Run Small LLM MedScore evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python run_small_llm_evaluation.py data/AskDocs.demo.jsonl results/
  
  # With baseline comparison
  python run_small_llm_evaluation.py data/AskDocs.demo.jsonl results/ \\
    --baseline MedScore_baseline_results/GPT4o_medscore_output.jsonl
  
  # Custom models
  python run_small_llm_evaluation.py data/AskDocs.demo.jsonl results/ \\
    --decomposition-model llama3.2:1b \\
    --verification-model llama3.2:1b
        """
    )

    parser.add_argument("input_file", help="Input JSONL file with medical responses")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--baseline", help="Path to baseline results for comparison")
    parser.add_argument("--decomposition-model", default="llama3.2:3b",
                        help="Model name for decomposition (default: llama3.2:3b)")
    parser.add_argument("--verification-model", default="llama3.2:3b",
                        help="Model name for verification (default: llama3.2:3b)")
    parser.add_argument("--reasoning-steps", type=int, default=3,
                        help="Number of reasoning steps (default: 3)")
    parser.add_argument("--no-chain-of-thought", action="store_true",
                        help="Disable chain-of-thought prompting")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence threshold for verification (default: 0.7)")
    parser.add_argument("--no-visualizations", action="store_true",
                        help="Skip creating visualizations")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_small_llm_evaluation(
            input_file=args.input_file,
            output_dir=args.output_dir,
            decomposition_model=args.decomposition_model,
            verification_model=args.verification_model,
            baseline_file=args.baseline,
            create_visualizations=not args.no_visualizations,
            reasoning_steps=args.reasoning_steps,
            use_chain_of_thought=not args.no_chain_of_thought,
            confidence_threshold=args.confidence_threshold
        )

        print(f"\nEvaluation completed successfully!")
        print(f"Final MedScore: {results['final_score']:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
