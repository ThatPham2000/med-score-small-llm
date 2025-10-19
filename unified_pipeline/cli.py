import argparse
import sys
from pathlib import Path

from llm_provider import create_llm_provider
from unified_pipeline import create_pipeline


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True, type=str, help="Input JSONL file with medical responses")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for results")

    # LLM Provider options
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Model name (default: provider-specific default)"
    )
    parser.add_argument(
        "--api_key",
        help="API key for OpenAI (if provider is openai)"
    )

    # Configuration options
    parser.add_argument(
        "--auto_config",
        action="store_true",
        help="Enable automatic configuration based on query analysis"
    )

    # Reasoning strategy
    parser.add_argument(
        "--reasoning_strategy",
        choices=["cot", "tot"],
        help="Reasoning strategy (cot or tot)"
    )
    parser.add_argument(
        "--tot_max_depth",
        type=int,
        help="Tree of Thoughts max depth"
    )
    parser.add_argument(
        "--tot_branching_factor",
        type=int,
        help="Tree of Thoughts branching factor"
    )

    # Knowledge and tools
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Enable RAG (Retrieval Augmented Generation)"
    )
    parser.add_argument(
        "--rag_top_k",
        type=int,
        help="Number of top documents to retrieve"
    )
    parser.add_argument(
        "--enable_tools",
        action="store_true",
        help="Enable external tools"
    )
    parser.add_argument(
        "--max_tool_steps",
        type=int,
        help="Maximum number of tool execution steps"
    )

    # Specialized features
    parser.add_argument(
        "--atomic_facts",
        action="store_true",
        help="Enable atomic fact decomposition"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation (0.0-1.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def load_queries_from_file(file_path: Path) -> list:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries
    except Exception as e:
        print(f"Error loading queries from {file_path}: {e}")
        sys.exit(1)


def create_pipeline_from_args(args):
    # Create LLM provider
    try:
        llm = create_llm_provider(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key
        )
    except Exception as e:
        print(f"Error creating LLM provider: {e}")
        sys.exit(1)

    try:
        pipeline = create_pipeline(
            llm=llm,
            auto_config=args.auto_config,
            reasoning_strategy=args.reasoning_strategy,
            tot_max_depth=args.tot_max_depth,
            tot_branching_factor=args.tot_branching_factor,
            use_rag=args.use_rag,
            rag_top_k=args.rag_top_k,
            enable_tools=args.enable_tools,
            max_tool_steps=args.max_tool_steps,
            enable_atomic_fact_decomposition=args.atomic_facts,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        return pipeline
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        sys.exit(1)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Create pipeline
    pipeline = create_pipeline_from_args(args)

    results = []

    # Load queries
    input_path = Path(args.input_file)
    queries = load_queries_from_file(input_path)

    # Process each query
    for query in queries:
        try:
            result = pipeline.process(query)
            results.append({
                "query": query,
                "result": result
            })
            if args.verbose:
                print(f"Processed query: {query}\nResult: {result}\n")
        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    # Save results to output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "results.jsonl"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(f"{res}\n")
        if args.verbose:
            print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
