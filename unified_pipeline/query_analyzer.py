from dataclasses import dataclass
from typing import Dict, Any

from .llm_provider import LLMProvider


@dataclass
class QueryCharacteristics:
    query_type: str = "general"  # general, mathematical, factual, creative, analytical, atomic_fact_decomposition
    complexity: str = "simple"  # simple, moderate, complex
    needs_computation: bool = False
    needs_knowledge: bool = False
    needs_exploration: bool = False
    confidence: float = 0.0
    reasoning: str = ""  # LLM's reasoning (if used)


class QueryAnalyzer:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze(self, query: str) -> QueryCharacteristics:
        return self._llm_analysis(query)

    def _llm_analysis(self, query: str) -> QueryCharacteristics:
        prompt = f"""Analyze this query and determine its characteristics.

Query: {query}

Analyze:
1. Query Type: Is it mathematical, factual, creative, analytical, or atomic_fact_decomposition?
2. Complexity: simple, moderate, or complex?
3. Requirements:
   - Does it need computation/calculations?
   - Does it need external knowledge/facts?
   - Does it need exploration of multiple approaches?

Special attention to atomic fact decomposition:
- "Please breakdown the following sentence into independent facts:" or similar prompts indicate atomic fact decomposition.

Respond in JSON format:
{{
    "query_type": "mathematical|factual|creative|analytical|atomic_fact_decomposition",
    "complexity": "simple|moderate|complex",
    "needs_computation": true|false,
    "needs_knowledge": true|false,
    "needs_exploration": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

        response = self.llm.generate_json(prompt, temperature=0.3)

        return QueryCharacteristics(
            query_type=response.get("query_type", "general"),
            complexity=response.get("complexity", "simple"),
            needs_computation=response.get("needs_computation", False),
            needs_knowledge=response.get("needs_knowledge", False),
            needs_exploration=response.get("needs_exploration", False),
            confidence=float(response.get("confidence", 0.0)),
            reasoning=response.get("reasoning", "LLM-based analysis")
        )

    def auto_configure(self, characteristics: QueryCharacteristics) -> Dict[str, Any]:
        config = {}

        # Atomic fact decomposition detection
        config["enable_atomic_fact_decomposition"] = (
                characteristics.query_type == "atomic_fact_decomposition"
        )

        # Reasoning strategy - always use CoT
        config["reasoning_strategy"] = "cot"

        # Tools: Enable for computation (code execution) OR exploration (web search)
        # For atomic fact decomposition, we typically don't need tools as we focus on text analysis
        config["enable_tools"] = (characteristics.needs_computation or characteristics.needs_exploration) and not \
        config["enable_atomic_fact_decomposition"]
        if config["enable_tools"]:
            config["max_tool_steps"] = 10

        # RAG
        config["use_rag"] = characteristics.needs_knowledge and not config["enable_atomic_fact_decomposition"]

        # Temperature
        if characteristics.query_type == "mathematical":
            config["temperature"] = 0.3  # Low for precision
        elif characteristics.query_type == "creative":
            config["temperature"] = 0.8  # High for creativity
        elif characteristics.query_type == "atomic_fact_decomposition":
            config["temperature"] = 0.3  # Low for precise fact extraction
        else:
            config["temperature"] = 0.7  # Balanced (general, factual, analytical)

        return config


def main():
    from .llm_provider import create_llm_provider
    import time

    try:
        llm = create_llm_provider("ollama", "llama3.2:3b")
        print("✅ Connected to Ollama")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    analyzer = QueryAnalyzer(llm)

    test_cases = [
        # Mathematical queries (needs_computation=True)
        {
            "query": "Calculate 15% of 240",
            "category": "Mathematical",
            "expected_chars": {
                "query_type": "mathematical",
                "complexity": "simple",
                "needs_computation": True,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.3
            }
        },
        {
            "query": "Solve the quadratic equation x² + 5x + 6 = 0 and explain each step",
            "category": "Mathematical Complex",
            "expected_chars": {
                "query_type": "mathematical",
                "complexity": "moderate",
                "needs_computation": True,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.3
            }
        },

        # Factual queries (needs_knowledge=True)
        {
            "query": "What is the capital of France?",
            "category": "Factual",
            "expected_chars": {
                "query_type": "factual",
                "complexity": "simple",
                "needs_computation": False,
                "needs_knowledge": True,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": True,
                "temperature": 0.7
            }
        },
        {
            "query": "Who invented the telephone and when was it first demonstrated?",
            "category": "Factual Complex",
            "expected_chars": {
                "query_type": "factual",
                "complexity": "moderate",
                "needs_computation": False,
                "needs_knowledge": True,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": True,
                "temperature": 0.7
            }
        },

        # Creative queries (needs_exploration=True)
        {
            "query": "Write a haiku about nature",
            "category": "Creative",
            "expected_chars": {
                "query_type": "creative",
                "complexity": "simple",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": True
            },
            "expected_config": {
                "reasoning_strategy": "tot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.8
            }
        },
        {
            "query": "Design a comprehensive marketing strategy for a new eco-friendly product targeting millennials",
            "category": "Creative Complex",
            "expected_chars": {
                "query_type": "creative",
                "complexity": "complex",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": True
            },
            "expected_config": {
                "reasoning_strategy": "tot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.8
            }
        },

        # Analytical queries (needs_exploration=True)
        {
            "query": "Compare Python vs JavaScript",
            "category": "Analytical",
            "expected_chars": {
                "query_type": "analytical",
                "complexity": "simple",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": True
            },
            "expected_config": {
                "reasoning_strategy": "tot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.7
            }
        },
        {
            "query": "Analyze the economic impact of remote work on urban development, considering housing markets, transportation, and local businesses",
            "category": "Analytical Complex",
            "expected_chars": {
                "query_type": "analytical",
                "complexity": "complex",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": True
            },
            "expected_config": {
                "reasoning_strategy": "tot",
                "enable_tools": True,
                "use_rag": False,
                "temperature": 0.7
            }
        },

        # General queries (no specific needs)
        {
            "query": "Hello, how are you?",
            "category": "General",
            "expected_chars": {
                "query_type": "general",
                "complexity": "simple",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": False,
                "temperature": 0.7
            }
        },

        # Atomic Fact Decomposition queries
        {
            "query": "Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.",
            "category": "Atomic Fact Decomposition",
            "expected_chars": {
                "query_type": "atomic_fact_decomposition",
                "complexity": "moderate",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": False,
                "enable_atomic_fact_decomposition": True,
                "temperature": 0.3
            }
        },
        {
            "query": "Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.",
            "category": "Atomic Fact Decomposition - Medical",
            "expected_chars": {
                "query_type": "atomic_fact_decomposition",
                "complexity": "simple",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": False,
                "enable_atomic_fact_decomposition": True,
                "temperature": 0.3
            }
        },
        {
            "query": "Extract verifiable facts from this medical text: The doctor believes that the likely sequence of events is that the COVID-19 infection led to demand ischemia, which in turn caused the myocardial infarction (MI).",
            "category": "Atomic Fact Decomposition - Complex Medical",
            "expected_chars": {
                "query_type": "atomic_fact_decomposition",
                "complexity": "complex",
                "needs_computation": False,
                "needs_knowledge": False,
                "needs_exploration": False
            },
            "expected_config": {
                "reasoning_strategy": "cot",
                "enable_tools": False,
                "use_rag": True,
                "enable_atomic_fact_decomposition": True,
                "temperature": 0.3
            }
        }
    ]

    print(f"\n📊 Testing {len(test_cases)} Comprehensive Scenarios")
    print("=" * 80)

    # Track results for summary
    results = {"correct": 0, "partial": 0, "incorrect": 0}

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        category = case["category"]
        expected_chars = case["expected_chars"]
        expected_config = case["expected_config"]

        print(f"\n[{i:2d}] {category}")
        print(f"     Query: {query}")

        # Analyze query
        start_time = time.time()
        chars = analyzer.analyze(query)
        config = analyzer.auto_configure(chars)
        elapsed = (time.time() - start_time) * 1000

        # Display comprehensive results
        print(f"     📋 QueryCharacteristics:")
        print(
            f"        Type: {chars.query_type:12} | Complexity: {chars.complexity:8} | Confidence: {chars.confidence:.2f}")
        print(
            f"        Needs: computation={chars.needs_computation}, knowledge={chars.needs_knowledge}, exploration={chars.needs_exploration}")
        print(f"        Reasoning: {chars.reasoning}")

        print(f"     ⚙️  Auto-configure:")
        print(
            f"        Strategy: {config['reasoning_strategy'].upper():4} | Tools: {config.get('enable_tools', False):5} | RAG: {config.get('use_rag', False):5}")
        if config.get('enable_atomic_fact_decomposition'):
            print(f"        Atomic Fact Decomposition: {config.get('enable_atomic_fact_decomposition', False)}")
        print(
            f"        Temperature: {config.get('temperature', 0.7)} | Max Tool Steps: {config.get('max_tool_steps', 0)}")
        if config.get('use_rag'):
            print(f"        RAG Top-K: {config.get('rag_top_k', 0)}")
        print(f"     ⏱️  Time: {elapsed:.0f}ms")

        # Verify characteristics
        char_matches = []
        if chars.query_type == expected_chars["query_type"]:
            char_matches.append("✅Type")
        if chars.complexity == expected_chars["complexity"]:
            char_matches.append("✅Complexity")
        if chars.needs_computation == expected_chars["needs_computation"]:
            char_matches.append("✅Computation")
        if chars.needs_knowledge == expected_chars["needs_knowledge"]:
            char_matches.append("✅Knowledge")
        if chars.needs_exploration == expected_chars["needs_exploration"]:
            char_matches.append("✅Exploration")

        # Verify configuration
        config_matches = []
        if config.get('reasoning_strategy') == expected_config["reasoning_strategy"]:
            config_matches.append("✅Strategy")
        if config.get('enable_tools') == expected_config["enable_tools"]:
            config_matches.append("✅Tools")
        if config.get('use_rag') == expected_config["use_rag"]:
            config_matches.append("✅RAG")
        if abs(config.get('temperature', 0.7) - expected_config["temperature"]) < 0.1:
            config_matches.append("✅Temp")
        if 'enable_atomic_fact_decomposition' in expected_config:
            if config.get('enable_atomic_fact_decomposition') == expected_config["enable_atomic_fact_decomposition"]:
                config_matches.append("✅AtomicFact")

        total_matches = len(char_matches) + len(config_matches)
        max_matches = 9  # 5 characteristics + 4 config items (including atomic fact decomposition)

        if total_matches == max_matches:
            results["correct"] += 1
            status = "✅ Perfect"
        elif total_matches >= 6:
            results["partial"] += 1
            status = "⚠️  Partial"
        else:
            results["incorrect"] += 1
            status = "❌ Incorrect"

        print(f"     Result: {status} ({total_matches}/{max_matches})")
        print(f"        Characteristics: {', '.join(char_matches) if char_matches else 'None'}")
        print(f"        Configuration: {', '.join(config_matches) if config_matches else 'None'}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    total = len(test_cases)
    print(f"Total Tests: {total}")
    print(f"✅ Perfect: {results['correct']} ({results['correct'] / total * 100:.1f}%)")
    print(f"⚠️  Partial: {results['partial']} ({results['partial'] / total * 100:.1f}%)")
    print(f"❌ Incorrect: {results['incorrect']} ({results['incorrect'] / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
