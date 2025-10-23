import re
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .llm_provider import LLMProvider


@dataclass
class ThoughtNode:
    """Represents a node in the Tree of Thoughts"""
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    evaluation_score: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_path(self) -> List[str]:
        """Get the reasoning path from root to this node as content strings"""
        path = []
        current = self
        while current:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))

    def get_node_path(self) -> List['ThoughtNode']:
        """Get the reasoning path from root to this node as ThoughtNode objects"""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))


class ChainOfThoughtReasoner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def reason_medical_atomic_facts_decompose(self):
        return {
            "reasoning_steps": [
                # Step 1: Identify main medical concepts
                "1. IDENTIFY MAIN MEDICAL CONCEPTS: First, identify the main medical concepts in the sentence - conditions, treatments, medications, procedures, or symptoms",
                
                # Step 2: Break down into verifiable facts
                "2. BREAK DOWN INTO VERIFIABLE FACTS: Then, break down each concept into verifiable facts that can be independently assessed",
                
                # Step 3: Ensure objectivity and verifiability
                "3. ENSURE OBJECTIVITY AND VERIFIABILITY: Next, ensure each fact is objective and can be verified against reliable medical sources",
                
                # Step 4: Validate completeness and context
                "4. VALIDATE COMPLETENESS AND CONTEXT: Then, validate that each fact is complete and contextually appropriate for medical assessment",
                
                # Step 5: MedScore Problem 1 - Filter unverifiable claims
                "5. FILTER UNVERIFIABLE CLAIMS: Remove personal narratives, patient-specific interactions, and bedside manner statements. Exclude: personal experiences ('I spoke with your doctor'), patient interactions, and subjective statements",
                
                # Step 6: MedScore Problem 2 - Prevent hallucinated claims
                "6. PREVENT HALLUCINATED CLAIMS: Ensure all claims are grounded in the original sentence. Verify no additional information beyond the sentence, no distortion of original meaning, and no irrelevant information",
                
                # Step 7: MedScore Problem 3 - Preserve complete claims
                "7. PRESERVE COMPLETE CLAIMS: Maintain important modifiers, conditional statements, temporal information, and cause-effect relationships. Ensure no critical medical details are lost",
                
                # Step 8: MedScore Problem 4 - Transform to declarative format
                "8. TRANSFORM TO DECLARATIVE FORMAT: Convert imperatives to declaratives, remove nested sub-clauses, and ensure declarative sentence structure. Transform 'They said [claim]' to '[claim]'",
                
                # Step 9: MedScore Problem 5 - Resolve context-dependent claims
                "9. RESOLVE CONTEXT-DEPENDENT CLAIMS: Replace ALL pronouns with specific entities from context: 'it/this/that/these/those' → specific medical terms, 'his/her/your/their' → specific persons, 'they/them' → specific medical professionals",
                
                # Step 10: MedScore Problem 6 - Eliminate redundant claims
                "10. ELIMINATE REDUNDANT CLAIMS: Remove duplicate or minimally different versions of the same fact. Focus on distinct, non-overlapping medical facts",
                
                # Step 11: MedScore Problem 7 - Ensure comprehensive coverage
                "11. ENSURE COMPREHENSIVE COVERAGE: Verify all important medical information is extracted. Don't miss key facts from the sentence. Maintain completeness of medical content",
                
                # Step 12: Review and refine for accuracy
                "12. REVIEW AND REFINE FOR ACCURACY: Review each extracted fact for: objectivity, verifiability, completeness, accuracy, and medical relevance. Ensure each fact can stand alone as an independent medical statement",
            ],
        }

    def reason(
            self,
            query: str,
            temperature: float = 0.7
    ) -> Dict[str, Any]:
        prompt = self._build_cot_prompt(query)
        response = self.llm.generate(prompt, temperature=temperature, max_tokens=2000)

        print('===============[Cot Response]\n', response)

        # Parse the response into reasoning steps only
        steps = self._parse_reasoning_steps(response)

        # The reasoning stage focuses only on the reasoning process
        # Final answers will be generated in later pipeline stages
        return {
            "query": query,
            "reasoning_steps": steps,
            "raw_output": response
        }

    def _build_cot_prompt(self, query: str) -> str:
        prompt = f"""You are an expert logical reasoning assistant. Analyze the following problem systematically and provide detailed step-by-step reasoning.

Problem: {query}

Instructions:
1. Read the problem carefully and identify all given information
2. Break down the problem into clear, logical reasoning steps
3. For each step, explain your thinking process clearly
4. Consider all possible scenarios and eliminate impossible ones
5. Use logical deduction and inference rules
6. For multiple choice questions, evaluate each option systematically
7. Show your work and reasoning for each step
8. Do NOT provide a final answer - focus only on the reasoning process

IMPORTANT: For logical reasoning problems:
- Identify all given conditions and constraints
- Use logical operators (if-then, and, or, not) correctly
- Consider all possible combinations and eliminate contradictions
- For spatial reasoning, visualize or map out the relationships
- For conditional logic, trace through all possible scenarios
- For argument analysis, identify premises, conclusions, and logical connections

Format your response as:

Step 1: [Your first reasoning step]
Step 2: [Your second reasoning step]
...

Begin your systematic reasoning:"""

        return prompt

    def _parse_reasoning_steps(self, response: str) -> List[str]:
        steps = []

        # Look for "Step X:" patterns
        step_pattern = r'Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|Final Answer:|$)'
        matches = re.finditer(step_pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_content = match.group(1).strip()
            if step_content:
                steps.append(step_content)

        return steps


class TreeOfThoughtsReasoner:
    """
    Tree of Thoughts (ToT) Reasoning
    Implements deliberate exploration with backtracking and self-evaluation
    """

    def __init__(
            self,
            llm: LLMProvider,
            max_depth: int = 5,
            branching_factor: int = 3,
            search_strategy: str = "bfs"  # "bfs" or "dfs"
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.search_strategy = search_strategy

    def reason(
            self,
            query: str,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate reasoning using Tree of Thoughts
        
        Returns the best reasoning path found through deliberate search
        """

        # Initialize root node
        root = ThoughtNode(
            content=f"Problem: {query}",
            depth=0
        )

        # Perform search
        if self.search_strategy == "bfs":
            best_path = self._breadth_first_search(root, query, temperature)
        else:
            best_path = self._depth_first_search(root, query, temperature)

        return {
            "query": query,
            "best_path": best_path,
            "reasoning_steps": [node.content for node in best_path[1:]],  # Skip root
            "final_answer": self._synthesize_answer(best_path, query, temperature),
            "explored_nodes": self._count_explored_nodes(root)
        }

    def _breadth_first_search(
            self,
            root: ThoughtNode,
            query: str,
            temperature: float
    ) -> List[ThoughtNode]:
        """BFS exploration of the thought tree"""

        queue = deque([root])
        best_path = [root]
        best_score = 0.0

        while queue:
            node = queue.popleft()

            # Stop if we've reached max depth
            if node.depth >= self.max_depth:
                continue

            # Generate candidate thoughts
            candidates = self._generate_thoughts(node, query, temperature)

            # Evaluate each candidate
            for thought_text in candidates:
                child = ThoughtNode(
                    content=thought_text,
                    parent=node,
                    depth=node.depth + 1
                )
                node.children.append(child)

                # Evaluate this thought
                child.evaluation_score = self._evaluate_thought(
                    child, query, temperature
                )

                # Check if this is a better path
                node_path = child.get_node_path()
                avg_score = sum(n.evaluation_score for n in node_path[1:]) / max(len(node_path) - 1, 1)

                if avg_score > best_score:
                    best_score = avg_score
                    best_path = node_path

                # Check if this is a terminal node (solution found)
                if child.evaluation_score >= 0.9 or self._is_solution(child, query, temperature):
                    child.is_terminal = True
                    return child.get_node_path()

                # Add to queue for further exploration
                queue.append(child)

        return best_path

    def _depth_first_search(
            self,
            root: ThoughtNode,
            query: str,
            temperature: float
    ) -> List[ThoughtNode]:
        """DFS exploration of the thought tree"""

        best_path = [root]
        best_score = 0.0

        def dfs_recursive(node: ThoughtNode):
            nonlocal best_path, best_score

            if node.depth >= self.max_depth:
                return

            # Generate and evaluate thoughts
            candidates = self._generate_thoughts(node, query, temperature)

            for thought_text in candidates:
                child = ThoughtNode(
                    content=thought_text,
                    parent=node,
                    depth=node.depth + 1
                )
                node.children.append(child)

                child.evaluation_score = self._evaluate_thought(
                    child, query, temperature
                )

                # Update best path if this is better
                node_path = child.get_node_path()
                avg_score = sum(n.evaluation_score for n in node_path[1:]) / max(len(node_path) - 1, 1)

                if avg_score > best_score:
                    best_score = avg_score
                    best_path = node_path

                # Check for terminal condition
                if child.evaluation_score >= 0.9 or self._is_solution(child, query, temperature):
                    child.is_terminal = True
                    return

                # Continue DFS
                dfs_recursive(child)

                if child.is_terminal:
                    return

        dfs_recursive(root)
        return best_path

    def _generate_thoughts(
            self,
            node: ThoughtNode,
            query: str,
            temperature: float
    ) -> List[str]:
        """Generate multiple candidate next thoughts"""

        current_path = "\n".join(node.get_path())

        prompt = f"""You are exploring solutions to a problem. Generate {self.branching_factor} different possible next steps.

Problem: {query}

Current reasoning path:
{current_path}

Generate {self.branching_factor} distinct and creative next steps. Each should explore a different approach or aspect of the problem.

Respond in JSON format:
{{"thoughts": ["thought 1", "thought 2", "thought 3"]}}"""

        try:
            response = self.llm.generate_json(prompt, temperature=temperature, max_tokens=10000)
            thoughts = response.get("thoughts", [])
            # Ensure all thoughts are strings, not dicts
            thoughts = [str(t) if not isinstance(t, str) else t for t in thoughts]
            return thoughts[:self.branching_factor]
        except:
            # Fallback: generate thoughts one at a time
            thoughts = []
            for i in range(self.branching_factor):
                thought_prompt = f"{prompt}\n\nGenerate thought #{i + 1}:"
                thought = self.llm.generate(thought_prompt, temperature=temperature, max_tokens=10000)
                thoughts.append(str(thought))
            return thoughts

    def _evaluate_thought(
            self,
            node: ThoughtNode,
            query: str,
            temperature: float
    ) -> float:
        """Evaluate the promise of a thought node"""

        current_path = "\n".join(node.get_path())

        prompt = f"""Evaluate the following reasoning path for solving a problem.

Problem: {query}

Reasoning path:
{current_path}

Rate this reasoning path on a scale of 0.0 to 1.0 based on:
- Logical coherence
- Progress toward the solution
- Correctness of reasoning
- Completeness

Respond in JSON format:
{{"score": 0.75, "reasoning": "explanation"}}"""

        try:
            response = self.llm.generate_json(prompt, temperature=0.3, max_tokens=10000)
            score = float(response.get("score", 0.5))
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except:
            # Fallback: simple heuristic based on length and keywords
            return 0.5

    def _is_solution(
            self,
            node: ThoughtNode,
            query: str,
            temperature: float
    ) -> bool:
        """Check if the current node represents a complete solution"""

        # Simple heuristic: check if depth is reasonable and content mentions conclusion
        if node.depth < 2:
            return False

        conclusion_keywords = ['therefore', 'thus', 'answer is', 'solution is', 'final']
        content_lower = node.content.lower()

        return any(keyword in content_lower for keyword in conclusion_keywords)

    def _synthesize_answer(
            self,
            path: List[ThoughtNode],
            query: str,
            temperature: float
    ) -> str:
        """Synthesize a final answer from the best path"""

        reasoning_trace = "\n\n".join([node.content for node in path[1:]])

        prompt = f"""Based on the following reasoning trace, provide a clear and concise final answer.

Problem: {query}

Reasoning trace:
{reasoning_trace}

Provide the final answer:"""

        return self.llm.generate(prompt, temperature=temperature, max_tokens=10000)

    def _count_explored_nodes(self, root: ThoughtNode) -> int:
        """Count total nodes explored"""
        count = 1
        for child in root.children:
            count += self._count_explored_nodes(child)
        return count
