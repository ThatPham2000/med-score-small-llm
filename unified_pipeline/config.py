FAST_CONFIG = {
    "auto_config": False,
    "reasoning_strategy": "cot",
    "use_rag": False,
    "enable_tools": False,
    "enable_atomic_fact_decomposition": False,
    "temperature": 0.7,
    "verbose": True
}

BALANCED_CONFIG = {
    "auto_config": True,
    "reasoning_strategy": "tot",
    "use_rag": True,
    "tot_max_depth": 3,
    "tot_branching_factor": 2,
    "enable_tools": True,
    "max_tool_steps": 5,
    "enable_atomic_fact_decomposition": False,
    "temperature": 0.7,
    "verbose": True
}

PREMIUM_CONFIG = {
    "auto_config": True,
    "reasoning_strategy": "tot",
    "use_rag": True,
    "tot_max_depth": 5,
    "tot_branching_factor": 3,
    "enable_tools": True,
    "max_tool_steps": 15,
    "enable_atomic_fact_decomposition": False,
    "temperature": 0.7,
    "verbose": True
}

# For mathematical reasoning
MATH_CONFIG = {
    "auto_config": False,
    "reasoning_strategy": "cot",  # CoT + tools for mathematical precision
    "use_rag": False,
    "enable_tools": True,
    "max_tool_steps": 10,
    "enable_atomic_fact_decomposition": False,
    "temperature": 0.3,  # Lower temperature for precision
    "verbose": True
}

# For knowledge-intensive tasks
KNOWLEDGE_CONFIG = {
    "auto_config": True,
    "reasoning_strategy": "tot",
    "use_rag": True,
    "tot_max_depth": 4,
    "tot_branching_factor": 2,
    "enable_tools": False,
    "enable_atomic_fact_decomposition": False,
    "temperature": 0.5,
    "verbose": True
}

# For atomic fact decomposition tasks
ATOMIC_FACT_CONFIG = {
    "auto_config": False,
    "reasoning_strategy": "cot",
    "use_rag": False,
    "enable_tools": False,
    "enable_atomic_fact_decomposition": True,
    "temperature": 0.3,  # Lower temperature for precision
    "verbose": True
}
