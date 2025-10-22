"""
Unified Pipeline: Optimized Four-Stage Reasoning System

A streamlined cascade of four interconnected stages:
1. Reasoning Core - Structure the problem with CoT or ToT
2. Knowledge Grounding - Retrieve and integrate external knowledge
3. Capability Augmentation - Use tools for computation and real-time data
4. Evaluation & Verification - Synthesize and verify the final answer
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .agentic_tools import AgenticFramework
from .knowledge_grounding import SimpleRetriever
from .llm_provider import LLMProvider
from .query_analyzer import QueryAnalyzer
from .reasoning_strategies import ChainOfThoughtReasoner, TreeOfThoughtsReasoner


@dataclass
class PipelineConfig:
    # use llm for generate config
    auto_config: bool = True

    reasoning_strategy: str = "cot"  # "cot" or "tot"
    tot_max_depth: int = 3
    tot_branching_factor: int = 2

    use_rag: bool = False
    rag_top_k: int = 5

    enable_tools: bool = False
    max_tool_steps: int = 10

    enable_atomic_fact_decomposition: bool = False

    temperature: float = 0.3
    verbose: bool = False


class UnifiedPipeline:
    def __init__(
            self,
            llm: LLMProvider,
            retriever: Optional[SimpleRetriever] = None,
            config: Optional[PipelineConfig] = None
    ):
        self.llm = llm
        self.config = config or PipelineConfig()

        self.cot_reasoner = ChainOfThoughtReasoner(llm)
        self.tot_reasoner = None  # Lazy initialization

        self.retriever = retriever or SimpleRetriever()

        self.agentic = AgenticFramework(llm)

    def process(self, query: str) -> Dict[str, Any]:
        start_time = time.time()

        # Auto-configure if enabled
        if self.config.auto_config:
            self._auto_configure(query)

        if self.config.verbose:
            print(f"\n{'=' * 10}\nProcessing: {query}\n{'=' * 10}")
            print(f"\n{'=' * 10}\nStage 1: Reasoning\n{'=' * 10}")

        # Stage 1: Reasoning
        reasoning_trace = self._reason(query)

        if self.config.verbose and self.config.use_rag:
            print(f"\n{'=' * 10}\nStage 2: Knowledge grounding\n{'=' * 10}")
        # Stage 2: Knowledge Grounding (if enabled)
        evidence = self._ground_knowledge(query) if self.config.use_rag else []

        if self.config.verbose and self.config.enable_tools:
            print(f"\n{'=' * 10}\nStage 3: Tool Augmentation\n{'=' * 10}")
        # Stage 3: Tool Augmentation (if enabled)
        tool_results = self._augment_with_tools(query, reasoning_trace,
                                                evidence) if self.config.enable_tools else []

        if self.config.verbose:
            print(f"\n{'=' * 10}\nStage 4: Synthesis & Verification\n{'=' * 10}")
        # Stage 4: Synthesize and Verify
        final_answer = self._synthesize_answer(query, reasoning_trace, evidence, tool_results)

        end_time = time.time()

        if self.config.verbose:
            print(f"\nAnswer: {final_answer}\nTime: {end_time - start_time:.2f}s\n{'=' * 10}")

        return {
            "final_answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "evidence": evidence,
            "tool_results": tool_results,
            "metadata": {
                "strategy": self.config.reasoning_strategy,
                "used_rag": bool(evidence),
                "used_tools": bool(tool_results),
                "atomic_fact_decomposition": self.config.enable_atomic_fact_decomposition,
                "processing_time": end_time - start_time
            }
        }

    def _auto_configure(self, query: str):
        """Stage 0: Auto-configure pipeline based on LLM query analysis"""
        # Create analyzer (always uses LLM)
        analyzer = QueryAnalyzer(self.llm)
        characteristics = analyzer.analyze(query)
        config_params = analyzer.auto_configure(characteristics)

        # Update configuration
        self.config.reasoning_strategy = config_params.get('reasoning_strategy', 'cot')
        self.config.use_rag = config_params.get('use_rag', False)
        self.config.enable_tools = config_params.get('enable_tools', False)
        self.config.enable_atomic_fact_decomposition = config_params.get('enable_atomic_fact_decomposition', False)
        self.config.temperature = config_params.get('temperature', 0.7)

        if self.config.reasoning_strategy == 'tot':
            self.config.tot_max_depth = config_params.get('tot_max_depth', 3)
            self.config.tot_branching_factor = config_params.get('tot_branching_factor', 2)
        if self.config.enable_tools:
            self.config.max_tool_steps = config_params.get('max_tool_steps', 10)

        if self.config.verbose:
            print(f"\n{'=' * 10}\nAuto-configured\n{'=' * 10}")
            print(f'Strategy: {self.config.reasoning_strategy.upper()}')
            print(f'Use RAG: {"Yes" if self.config.use_rag else "No"}')
            print(f'Enable Tools: {"Yes" if self.config.enable_tools else "No"}')
            print(f'Atomic Fact Decomposition: {"Yes" if self.config.enable_atomic_fact_decomposition else "No"}')

    def _reason(self, query: str) -> list:
        """Stage 1: Apply reasoning strategy"""
        if self.config.reasoning_strategy == "tot":
            # Lazy initialize ToT reasoner
            if self.tot_reasoner is None:
                self.tot_reasoner = TreeOfThoughtsReasoner(
                    llm=self.llm,
                    max_depth=self.config.tot_max_depth,
                    branching_factor=self.config.tot_branching_factor
                )
            result = self.tot_reasoner.reason(query,
                                              self.config.temperature)  # TODO(THAT): check for atomic fact decomposition
            return result.get("reasoning_steps", [])
        else:
            # Default to CoT
            result = self.cot_reasoner.reason_medical_atomic_facts_decompose() if self.config.enable_atomic_fact_decomposition else self.cot_reasoner.reason(
                query, temperature=self.config.temperature)
            return result.get("reasoning_steps", [])

    def _ground_knowledge(self, query: str) -> list:
        """Stage 2: Retrieve relevant evidence"""
        evidence = self.retriever.retrieve(query, top_k=self.config.rag_top_k)
        if self.config.verbose and evidence:
            print(f"Retrieved {len(evidence)} evidence pieces")
        return evidence

    def _augment_with_tools(self, query: str, reasoning: list, evidence: list) -> list:
        """Stage 3: Use external tools if needed"""
        # Build enhanced context with reasoning from Stage 1 and evidence
        context_parts = []

        # Add reasoning from Stage 1 to provide continuity
        if reasoning:
            reasoning_text = "\n".join([
                f"Step {i + 1}: {step}" if isinstance(step, str) else str(step)
                for i, step in enumerate(reasoning)
            ])
            context_parts.append(f"Previous reasoning:\n{reasoning_text}")

        # Add evidence from Stage 2
        if evidence:
            evidence_text = "\n".join([f"[{e.source}] {e.content}" for e in evidence])
            context_parts.append(f"Evidence:\n{evidence_text}")

        # Combine all context parts
        aug_context = "\n\n".join(context_parts).strip()

        result = self.agentic.reason_with_tools(
            query,
            context=aug_context,
            max_steps=self.config.max_tool_steps,
            temperature=self.config.temperature
        )

        tool_calls = result.get("tool_calls", [])
        if self.config.verbose and tool_calls:
            print(f"Executed {len(tool_calls)} tool calls")

        return tool_calls

    def _synthesize_answer(self, query: str, reasoning: list, evidence: list, tools: list) -> str:
        """Stage 4: Synthesize final answer from all sources"""

        # Build comprehensive context
        context_parts = []

        if reasoning:
            reasoning_text = "\n".join([
                f"Step {i + 1}: {step}" if isinstance(step, str) else str(step)
                for i, step in enumerate(reasoning)
            ])
            context_parts.append(f"Reasoning:\n{reasoning_text}")

        if evidence:
            evidence_text = "\n".join([
                f"[{e.source}] {e.content}" for e in evidence[:3]  # Limit to 3 pieces
            ])
            context_parts.append(f"Evidence:\n{evidence_text}")

        if tools:
            tool_text = "\n".join([
                f"{t.get('tool', 'unknown')}: {t.get('result', {}).output if hasattr(t.get('result', {}), 'output') else 'N/A'}"
                for t in tools[:3]  # Limit to 3 tool results
            ])
            context_parts.append(f"Tool Results:\n{tool_text}")

        full_context = "\n\n".join(context_parts)

        if self.config.enable_atomic_fact_decomposition:
            # Use chat with system prompt for atomic fact decomposition
            messages = [
                {"role": "system",
                 "content": self._build_medical_atomic_fact_decomposition_system_prompt(full_context)},
                {"role": "user", "content": query}
            ]
            answer = self.llm.chat(messages, temperature=self.config.temperature, max_tokens=10000)
        else:
            # Optimized prompt for logical reasoning and multiple choice questions
            prompt = f"""You are an expert logical reasoning assistant. Based on the detailed reasoning provided, give the final answer to the query.

CRITICAL INSTRUCTIONS:
- For multiple choice questions, respond with EXACTLY the chosen option text
- Do not add any extra words, explanations, or formatting
- If the reasoning leads to a clear conclusion, state it directly
- If uncertain, choose the most logically sound option based on the reasoning

Reasoning Analysis:
{full_context}

Query:
{query}

Based on the reasoning above, provide the final answer:"""
            answer = self.llm.generate(prompt, temperature=self.config.temperature, max_tokens=10000)
        return answer.strip()

    def _build_medical_atomic_fact_decomposition_system_prompt(self, full_context: str) -> str:
        """Full context in this case is the reasoning steps."""
        return f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning while addressing the 7 MedScoreTaxonomy issues.

CRITICAL INSTRUCTIONS:
- Focus on ONE medical concept per claim
- Use simple, declarative sentences
- Avoid complex medical jargon when possible
- Extract facts that can be verified independently
- Preserve all important medical modifiers and conditions

REASONING FORMAT:
Think step by step:
{full_context}

If there is no verifiable fact in the sentence, please write "No verifiable claim".

HERE ARE SOME EXAMPLES WITH REASONING:
Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. \n\n Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.\n\nYour doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
1. IDENTIFY MAIN MEDICAL CONCEPTS: doctor consultation, tetanus concerns
2. BREAK DOWN INTO VERIFIABLE FACTS: None, this is a personal narrative about speaking to a doctor
3. ENSURE OBJECTIVITY AND VERIFIABILITY: No, this is a subjective personal experience
4. VALIDATE COMPLETENESS AND CONTEXT: N/A, no verifiable facts
5. FILTER UNVERIFIABLE CLAIMS: This is a personal narrative ("I spoke to your doctor") - should be excluded
6. PREVENT HALLUCINATED CLAIMS: N/A, no verifiable claims to assess
7. PRESERVE COMPLETE CLAIMS: N/A, no verifiable claims
8. TRANSFORM TO DECLARATIVE FORMAT: N/A, no verifiable claims
9. RESOLVE CONTEXT-DEPENDENT CLAIMS: N/A, no verifiable claims
10. ELIMINATE REDUNDANT CLAIMS: N/A, no verifiable claims
11. ENSURE COMPREHENSIVE COVERAGE: N/A, no verifiable claims
12. REVIEW AND REFINE FOR ACCURACY: N/A, no verifiable claims

Facts:
- No verifiable claim

Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.\n\nThe doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped.\n\nThe doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition.\n\nIt's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
1. IDENTIFY MAIN MEDICAL CONCEPTS: substances, muscle health, bone health, risks, side effects
2. BREAK DOWN INTO VERIFIABLE FACTS: Substances can have positive effects on muscle health, bone health, and carry risks and side effects
3. ENSURE OBJECTIVITY AND VERIFIABILITY: Yes, these are objective medical facts about substance effects that can be verified against medical literature
4. VALIDATE COMPLETENESS AND CONTEXT: Yes, they are complete and contextually appropriate for medical assessment
5. FILTER UNVERIFIABLE CLAIMS: No personal narratives or subjective statements found - all claims are medical facts
6. PREVENT HALLUCINATED CLAIMS: All claims are grounded in the original sentence, no additional information added
7. PRESERVE COMPLETE CLAIMS: Important modifiers like "may have" and "potential" are preserved
8. TRANSFORM TO DECLARATIVE FORMAT: Claims are already in declarative format
9. RESOLVE CONTEXT-DEPENDENT CLAIMS: "these substances" → "anabolic steroids" (from context), "they" → "anabolic steroids"
10. ELIMINATE REDUNDANT CLAIMS: Each fact addresses a distinct aspect (muscle health, bone health, risks, side effects)
11. ENSURE COMPREHENSIVE COVERAGE: All important medical information about substance effects is extracted
12. REVIEW AND REFINE FOR ACCURACY: Each fact is objective, verifiable, complete, and medically accurate

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids may carry significant risks.
- Anabolic steroids may carry potential side effects.

Context: The patient was diagnosed with diabetes and prescribed metformin. The doctor explained that this medication helps control blood sugar levels and that it should be taken with meals. They also mentioned that these tablets can cause gastrointestinal side effects, but those usually subside after a few weeks. The patient asked about insulin, and the doctor said that it might be needed later if this treatment doesn't work effectively.

Please breakdown the following sentence into independent facts: The doctor explained that this medication helps control blood sugar levels and that it should be taken with meals.

Reasoning:
1. IDENTIFY MAIN MEDICAL CONCEPTS: medication, blood sugar control, administration timing
2. BREAK DOWN INTO VERIFIABLE FACTS: Metformin helps control blood sugar levels, metformin should be taken with meals
3. ENSURE OBJECTIVITY AND VERIFIABILITY: Yes, these are objective medical facts about metformin
4. VALIDATE COMPLETENESS AND CONTEXT: Yes, they are complete and contextually appropriate
5. FILTER UNVERIFIABLE CLAIMS: No personal narratives or subjective statements found
6. PREVENT HALLUCINATED CLAIMS: All claims are grounded in the original sentence
7. PRESERVE COMPLETE CLAIMS: Important modifiers and timing information preserved
8. TRANSFORM TO DECLARATIVE FORMAT: Claims are already in declarative format
9. RESOLVE CONTEXT-DEPENDENT CLAIMS: "this medication" → "metformin" (from context), "it" → "metformin"
10. ELIMINATE REDUNDANT CLAIMS: Each fact addresses a distinct aspect (efficacy, administration)
11. ENSURE COMPREHENSIVE COVERAGE: All important medical information about metformin is extracted
12. REVIEW AND REFINE FOR ACCURACY: Each fact is objective, verifiable, complete, and medically accurate

Facts:
- Metformin helps control blood sugar levels.
- Metformin should be taken with meals.

Context: I was prescribed antibiotics for their infection. The doctor explained that this medication should be taken twice daily for 7 days, and that it may cause nausea in some patients. They also mentioned that if the patient experiences severe side effects, they should contact the doctor immediately.

Please breakdown the following sentence into independent facts: The doctor explained that this medication should be taken twice daily for 7 days, and that it may cause nausea in some patients.

Reasoning:
1. IDENTIFY MAIN MEDICAL CONCEPTS: medication administration, side effects
2. BREAK DOWN INTO VERIFIABLE FACTS: Antibiotics should be taken twice daily for 7 days, antibiotics may cause nausea
3. ENSURE OBJECTIVITY AND VERIFIABILITY: Yes, these are objective medical facts about antibiotics
4. VALIDATE COMPLETENESS AND CONTEXT: Yes, they are complete and contextually appropriate
5. FILTER UNVERIFIABLE CLAIMS: No personal narratives or subjective statements found
6. PREVENT HALLUCINATED CLAIMS: All claims are grounded in the original sentence
7. PRESERVE COMPLETE CLAIMS: Important modifiers like "twice daily", "7 days", "may cause" preserved
8. TRANSFORM TO DECLARATIVE FORMAT: Claims are already in declarative format
9. RESOLVE CONTEXT-DEPENDENT CLAIMS: "This medication" → "antibiotics" (from context), "it" → "antibiotics"
10. ELIMINATE REDUNDANT CLAIMS: Each fact addresses a distinct aspect (administration, side effects)
11. ENSURE COMPREHENSIVE COVERAGE: All important medical information about antibiotics extracted
12. REVIEW AND REFINE FOR ACCURACY: Each fact is objective, verifiable, complete, and medically accurate

Facts:
- Antibiotics should be taken twice daily for 7 days.
- Antibiotics may cause nausea in some patients.

Now, for your task, follow the same reasoning process."""


# Convenience factory functions
def create_pipeline(
        llm: LLMProvider,
        retriever: Optional[SimpleRetriever] = None,
        auto_config: bool = False,
        reasoning_strategy: str = "cot",
        tot_max_depth: int = 3,
        tot_branching_factor: int = 2,
        use_rag: bool = False,
        rag_top_k: int = 5,
        enable_tools: bool = False,
        max_tool_steps: int = 10,
        enable_atomic_fact_decomposition: bool = False,
        temperature: float = 0.7,
        verbose: bool = False
) -> UnifiedPipeline:
    config = PipelineConfig(
        auto_config=auto_config,
        reasoning_strategy=reasoning_strategy,
        tot_max_depth=tot_max_depth,
        tot_branching_factor=tot_branching_factor,
        use_rag=use_rag,
        rag_top_k=rag_top_k,
        enable_tools=enable_tools,
        max_tool_steps=max_tool_steps,
        enable_atomic_fact_decomposition=enable_atomic_fact_decomposition,
        temperature=temperature,
        verbose=verbose
    )
    return UnifiedPipeline(llm, retriever, config)


# Main test function
def main():
    """Test the optimized pipeline"""
    from .llm_provider import create_llm_provider

    print("OPTIMIZED UNIFIED PIPELINE - Test")

    # Initialize
    print("\nInitializing LLM...")
    try:
        llm = create_llm_provider("ollama", "gemma3:12b")
        print("✅ LLM initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return

    # Create pipeline with auto-config
    pipeline = create_pipeline(llm, enable_atomic_fact_decomposition=True, verbose=True)
    # pipeline = create_pipeline(llm, auto_config=True, verbose=True)

    question1 = """Context: I spoke to your doctor and they recommended that you see a different doctor for a thorough assessment to determine the cause of your dropping oxygen levels. They believe that speculating about the possible causes without a proper evaluation wouldn't be helpful. However, they did mention that the upcoming sleep study is a good starting point. Additionally, other tests such as an echocardiogram or blood gas analysis may be necessary to help identify the underlying issue. Your doctor suggests that it's best to have a comprehensive evaluation to explore the various possibilities.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they recommended that you see a different doctor for a thorough assessment to determine the cause of your dropping oxygen levels.

IMPORTANT: Extract ONE medical concept per claim. Use simple, declarative sentences.

Facts:
"""
    question2 = """Context: I spoke to your doctor and they wanted to thank you for your interest in creating a language course to help physicians better communicate with patients who speak different languages. 

They mentioned that while language barriers can contribute to the "revolving door syndrome," it's just one of many factors. Other important factors include education, home support, medication noncompliance, and lack of primary care. 

In terms of a language course, your doctor thinks that Duolingo is a good option. However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish. They also mentioned that many Spanish-speaking patients have some knowledge of English or have family members who are fluent in English.

Your doctor didn't specify a preferred medium for the course, but they seemed to appreciate the idea of a convenient and accessible program. They also didn't provide specific vocabulary recommendations, but it's likely that a course focused on medical terminology and common patient interactions would be most useful.

Please breakdown the following sentence into independent facts: However, they noted that it's challenging for doctors to find the time to learn multiple languages, as there are many languages spoken by patients in their area, including Spanish, Hmong, Chinese, and Polish.

IMPORTANT: Extract ONE medical concept per claim. Use simple, declarative sentences.

Facts:
"""
    question3 = """Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. 

Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine.

Your doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

    Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

    IMPORTANT: Extract ONE medical concept per claim. Use simple, declarative sentences.

    Facts:
    """

    question4 = """If James drives to Canada at a speed of 60 mph and the distance is 360 miles, with a 1-hour stop along the way, how long will it take him to reach Canada?"""

    # Test queries
    test_queries = [
        question2,
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}===========:\n{query}")

        result = pipeline.process(query)

        print(f"\n✅ Answer: {result['final_answer']}")
        print(f"📊 Strategy: {result['metadata']['strategy']}")
        print(f"⏱️  Time: {result['metadata']['processing_time']:.2f}s")

    print("All tests completed!")


if __name__ == "__main__":
    main()
