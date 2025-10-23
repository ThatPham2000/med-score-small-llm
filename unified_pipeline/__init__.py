__version__ = "1.0.0"
__author__ = "distill_llm_to_slm"

from .agentic_tools import AgenticFramework, CodeExecutionTool, WebSearchTool, MindMapTool, Tool, ToolResult
from .config import FAST_CONFIG, BALANCED_CONFIG, PREMIUM_CONFIG, MATH_CONFIG, KNOWLEDGE_CONFIG
from .knowledge_grounding import SimpleRetriever, Evidence
from .llm_provider import create_llm_provider, LLMProvider, OpenAIProvider, OllamaProvider
from .query_analyzer import QueryAnalyzer, QueryCharacteristics
from .reasoning_strategies import ChainOfThoughtReasoner
from .unified_pipeline import (
    UnifiedPipeline,
    PipelineConfig,
    create_pipeline
)

__all__ = [
    # Core Pipeline
    "UnifiedPipeline",
    "PipelineConfig",
    "create_pipeline",

    # LLM Providers
    "create_llm_provider",
    "LLMProvider",
    "OpenAIProvider",
    "OllamaProvider",

    # Reasoning
    "ChainOfThoughtReasoner",

    # Knowledge Grounding
    "SimpleRetriever",
    "Evidence",

    # Agentic Tools
    "AgenticFramework",
    "CodeExecutionTool",
    "WebSearchTool",
    "MindMapTool",
    "Tool",
    "ToolResult",

    # Configuration
    "FAST_CONFIG",
    "BALANCED_CONFIG",
    "PREMIUM_CONFIG",
    "MATH_CONFIG",
    "KNOWLEDGE_CONFIG",

    # Query Analysis & Auto-Configuration
    "QueryAnalyzer",
    "QueryCharacteristics",
]
