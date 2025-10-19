import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from llm_provider import LLMProvider


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract Base Class for tools"""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    @abstractmethod
    def describe(self) -> str:
        pass


class CodeExecutionTool(Tool):
    """
    Tool for executing Python code
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str, **kwargs) -> ToolResult:
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute the code in a subprocess for isolation
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    return ToolResult(
                        success=True,
                        output=result.stdout.strip(),
                        metadata={"stderr": result.stderr}
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Execution error: {result.stderr}"
                    )

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Code execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution failed: {str(e)}"
            )

    def describe(self) -> str:
        return """Code Execution Tool: Execute Python code for precise calculations, data processing, and algorithmic operations.
        
Usage: Provide Python code as a string. The code will be executed in an isolated environment.
Example: code = "result = 123 * 456\\nprint(result)" """


class WebSearchTool(Tool):
    """
    Tool for searching the web (simulated for local use)
    In production, integrate with real search APIs (Google, Bing, DuckDuckGo)
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
        # In production, add API keys for real search services
        self.simulated = True

    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """
        Search the web for information
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            ToolResult with search results
        """

        if self.simulated:
            # Simulated search for demonstration
            # In production, replace with actual API calls
            results = self._simulated_search(query, num_results)
        else:
            # Real search implementation would go here
            results = []

        return ToolResult(
            success=True,
            output=results,
            metadata={"query": query, "num_results": len(results)}
        )

    def _simulated_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Simulate search results for demonstration"""

        # In a real implementation, this would call a search API
        # For now, return simulated results based on the query

        if self.llm:
            # Use LLM to generate simulated search results
            prompt = f"""Simulate web search results for the query: "{query}"

Generate {num_results} realistic search results with titles and snippets.

Respond in JSON format:
{{
    "results": [
        {{"title": "Result title", "snippet": "Brief description...", "url": "https://example.com"}},
        ...
    ]
}}"""

            try:
                response = self.llm.generate_json(prompt, temperature=0.7, max_tokens=10000)
                return response.get("results", [])
            except:
                pass

        # Fallback: generic results
        return [
            {
                "title": f"Search result {i + 1} for '{query}'",
                "snippet": f"This is a simulated search result snippet about {query}.",
                "url": f"https://example.com/result{i + 1}"
            }
            for i in range(num_results)
        ]

    def describe(self) -> str:
        return """Web Search Tool: Search the web for real-time information and current data.
        
Usage: Provide a search query string.
Example: query = "latest developments in quantum computing 2025" """


class MindMapTool(Tool):
    """
    Mind Map: Structured memory for tracking reasoning context
    Maintains a knowledge graph of entities and relationships
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, str]] = []

    def execute(self, action: str, **kwargs) -> ToolResult:
        """
        Interact with the mind map
        
        Args:
            action: "add_entity", "add_relationship", "query", or "get_all"
            
        Returns:
            ToolResult with mind map data
        """

        if action == "add_entity":
            return self._add_entity(**kwargs)
        elif action == "add_relationship":
            return self._add_relationship(**kwargs)
        elif action == "query":
            return self._query(**kwargs)
        elif action == "get_all":
            return self._get_all()
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown action: {action}"
            )

    def _add_entity(self, name: str, type: str, properties: Optional[Dict] = None) -> ToolResult:
        self.entities[name] = {
            "type": type,
            "properties": properties or {}
        }

        return ToolResult(
            success=True,
            output=f"Added entity: {name}",
            metadata={"entity": name}
        )

    def _add_relationship(self, from_entity: str, to_entity: str, relationship: str) -> ToolResult:
        self.relationships.append({
            "from": from_entity,
            "to": to_entity,
            "relationship": relationship
        })

        return ToolResult(
            success=True,
            output=f"Added relationship: {from_entity} -{relationship}-> {to_entity}",
            metadata={"relationship": relationship}
        )

    def _query(self, entity: Optional[str] = None, query: Optional[str] = None) -> ToolResult:
        if entity:
            # Return specific entity
            if entity in self.entities:
                # Find related entities
                related = []
                for rel in self.relationships:
                    if rel["from"] == entity:
                        related.append(f"{rel['relationship']} -> {rel['to']}")
                    elif rel["to"] == entity:
                        related.append(f"{rel['from']} -> {rel['relationship']}")

                return ToolResult(
                    success=True,
                    output={
                        "entity": self.entities[entity],
                        "relationships": related
                    }
                )
            else:
                return ToolResult(success=False, output=None, error=f"Entity not found: {entity}")

        elif query:
            # Use LLM to answer query based on mind map
            mind_map_str = self._format_mind_map()

            prompt = f"""Answer the following query based on the mind map:

Mind Map:
{mind_map_str}

Query: {query}

Answer:"""

            answer = self.llm.generate(prompt, temperature=0.5, max_tokens=10000)

            return ToolResult(success=True, output=answer)

        else:
            return self._get_all()

    def _get_all(self) -> ToolResult:
        return ToolResult(
            success=True,
            output={
                "entities": self.entities,
                "relationships": self.relationships
            }
        )

    def _format_mind_map(self) -> str:
        """Format mind map as readable text"""

        lines = ["Entities:"]
        for name, data in self.entities.items():
            lines.append(f"  - {name} ({data['type']})")

        lines.append("\nRelationships:")
        for rel in self.relationships:
            lines.append(f"  - {rel['from']} -{rel['relationship']}-> {rel['to']}")

        return "\n".join(lines)

    def describe(self) -> str:
        return """Mind Map Tool: Structured memory for tracking entities and relationships during reasoning.
        
Actions:
- add_entity: Add a new entity (name, type, properties)
- add_relationship: Add a relationship between entities (from_entity, to_entity, relationship)
- query: Query the mind map (entity or query)
- get_all: Get the entire mind map structure"""


class AgenticFramework:
    """
    Agentic Framework: Orchestrates tool use during reasoning
    Allows the LLM to determine when to use external tools
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.tools["code_execution"] = CodeExecutionTool()
        self.tools["web_search"] = WebSearchTool(llm=self.llm)
        self.tools["mind_map"] = MindMapTool(llm=self.llm)

    def reason_with_tools(
            self,
            query: str,
            context: str = "",
            max_steps: int = 10,
            temperature: float = 0.7
    ) -> Dict[str, Any]:
        reasoning_trace = []
        tool_calls = []

        for step in range(max_steps):
            print('Tools call=========:', tool_calls)

            # Determine next action
            action = self._plan_next_action(
                query, context, reasoning_trace, temperature
            )

            if action["type"] == "answer":
                # We have a final answer
                return {
                    "query": query,
                    "reasoning_trace": reasoning_trace,
                    "tool_calls": tool_calls,
                    "final_answer": action["content"],
                    "steps_taken": step + 1
                }

            elif action["type"] == "tool":
                tool_name = action.get("tool", "unknown")
                tool_params = action.get("parameters", {})

                # Ensure parameters is a dict
                if not isinstance(tool_params, dict):
                    tool_params = {}

                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].execute(**tool_params)
                    except TypeError as e:
                        # Handle missing required parameters
                        result = ToolResult(
                            success=False,
                            output=None,
                            error=f"Invalid parameters for {tool_name}: {str(e)}"
                        )

                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": tool_params,
                        "result": result
                    })

                    reasoning_trace.append({
                        "type": "tool_use",
                        "tool": tool_name,
                        "input": tool_params,
                        "output": result.output if result.success else result.error,
                        "success": result.success
                    })
                else:
                    reasoning_trace.append({
                        "type": "error",
                        "message": f"Tool not found: {tool_name}"
                    })

            elif action["type"] == "thought":
                # Regular reasoning step
                reasoning_trace.append({
                    "type": "thought",
                    "content": action["content"]
                })

        # Max steps reached
        return {
            "query": query,
            "reasoning_trace": reasoning_trace,
            "tool_calls": tool_calls,
            "final_answer": "Maximum reasoning steps reached without finding an answer.",
            "steps_taken": max_steps
        }

    def _plan_next_action(
            self,
            query: str,
            context: str,
            reasoning_trace: List[Dict[str, Any]],
            temperature: float
    ) -> Dict[str, Any]:
        """Plan the next action using the LLM"""

        # Format reasoning trace
        trace_str = self._format_trace(reasoning_trace)

        # Format available tools
        tools_str = self._format_tools()

        prompt = f"""You are solving a problem with access to external tools. Plan your next action.

Problem: {query}
{f"Context: {context}" if context else ""}

Reasoning trace so far:
{trace_str if trace_str else "No steps yet."}

Available tools:
{tools_str}

What should you do next? Choose one:
1. Use a tool (if you need external capabilities)
2. Make a reasoning step (if you need to think)
3. Provide the final answer (if you're done)

Respond in JSON format:
For tool use:
{{"type": "tool", "tool": "tool_name", "parameters": {{"param1": "value1"}}, "reasoning": "why"}}

For reasoning:
{{"type": "thought", "content": "your reasoning step"}}

For final answer:
{{"type": "answer", "content": "your final answer"}}

Your action:"""

        try:
            print(f'\n============PLAN NEXT ACTION PROMPT:\n{prompt}\n=================')
            response = self.llm.generate_json(prompt, temperature=temperature, max_tokens=10000)
            return response
        except Exception as e:
            # Fallback: just reason
            return {
                "type": "thought",
                "content": "Continuing to reason about the problem..."
            }

    def _format_trace(self, trace: List[Dict[str, Any]]) -> str:
        if not trace:
            return ""

        lines = []
        for i, step in enumerate(trace, 1):
            if step["type"] == "thought":
                lines.append(f"{i}. Thought: {step['content']}")
            elif step["type"] == "tool_use":
                status = "✓" if step["success"] else "✗"
                lines.append(f"{i}. Tool [{step['tool']}] {status}: {step['output']}")
            elif step["type"] == "error":
                lines.append(f"{i}. Error: {step['message']}")

        return "\n".join(lines)

    def _format_tools(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.describe()}")

        return "\n".join(lines)


# if __name__ == '__main__':
    # tool = CodeExecutionTool()
    # code = """
    # result = 123 + 456
    # print(result)
    #     """
    # result = tool.execute(code=code)
    # print("Code Execution Result:", result)
    #
    # print(f"\n{'-' * 80}\n")
    # tool = MindMapTool(llm=create_llm_provider(provider="ollama", model="gemma3:12b"))
    # res1 = tool.execute(action="add_entity", name="Python", type="Programming Language")
    # print(res1)
    # res2 = tool.execute(action="add_entity", name="JavaScript", type="Programming Language")
    # print(res2)
    # res3 = tool.execute(action="add_relationship", from_entity="Python", to_entity="JavaScript",
    #                     relationship="influences")
    # print(res3)
    # res4 = tool.execute(action="query", entity="Python")
    # print(res4)
    # res5 = tool.execute(action="query", query="What programming languages are influenced by Python?")
    # print(res5)
