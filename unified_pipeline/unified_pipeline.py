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
from .reasoning_strategies import ChainOfThoughtReasoner


@dataclass
class PipelineConfig:
    # use llm for generate config
    auto_config: bool = True

    reasoning_strategy: str = "cot"  # "cot" only

    use_rag: bool = False
    documents_path: str = ""

    enable_tools: bool = False
    max_tool_steps: int = 10

    enable_atomic_fact_decomposition: bool = False

    temperature: float = 0.3
    verbose: bool = False


class UnifiedPipeline:
    def __init__(
            self,
            llm: LLMProvider,
            config: Optional[PipelineConfig] = None
    ):
        self.llm = llm
        self.config = config or PipelineConfig()

        self.cot_reasoner = ChainOfThoughtReasoner(llm)

        self.retriever = SimpleRetriever(self.config.documents_path)

        self.agentic = AgenticFramework(llm)

    def process(self, query: str) -> Dict[str, Any]:
        start_time = time.time()

        # Auto-configure if enabled
        if self.config.auto_config:
            self._auto_configure(query)

        if self.config.verbose:
            print(f"\n{'=' * 10}\nProcessing: {query}\n{'=' * 10}")

        # Stage 1: Knowledge Grounding (if enabled)
        if self.config.verbose and self.config.use_rag:
            print(f"\n{'=' * 10}\nStage 1: Knowledge grounding\n{'=' * 10}")
        evidence = self._ground_knowledge(query) if self.config.use_rag else []

        # Stage 2: Reasoning
        if self.config.verbose:
            print(f"\n{'=' * 10}\nStage 2: Reasoning\n{'=' * 10}")
        reasoning_trace = self._reason(query, evidence)

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

        if self.config.enable_tools:
            self.config.max_tool_steps = config_params.get('max_tool_steps', 10)

        if self.config.verbose:
            print(f"\n{'=' * 10}\nAuto-configured\n{'=' * 10}")
            print(f'Strategy: {self.config.reasoning_strategy.upper()}')
            print(f'Use RAG: {"Yes" if self.config.use_rag else "No"}')
            print(f'Enable Tools: {"Yes" if self.config.enable_tools else "No"}')
            print(f'Atomic Fact Decomposition: {"Yes" if self.config.enable_atomic_fact_decomposition else "No"}')

    def _ground_knowledge(self, query: str) -> list:
        """Stage 1: Retrieve relevant evidence"""
        evidence = self.retriever.retrieve(query)
        if self.config.verbose and evidence:
            print(f"Retrieved {len(evidence)} evidence pieces")
        return evidence

    def _reason(self, query: str, evidence: list) -> list:
        """Stage 2: Apply reasoning strategy"""
        # Use CoT reasoning
        result = self.cot_reasoner.reason_medical_atomic_facts_decompose() if self.config.enable_atomic_fact_decomposition else self.cot_reasoner.reason(
            query, evidence, temperature=self.config.temperature)
        return result.get("reasoning_steps", [])

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
            evidence_text = "\n".join([f"[{e.content}" for e in evidence])
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
            evidence_text = "\n".join([f"{e.content}" for e in evidence])
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
- Review the entire reasoning process carefully once again
- Synthesize information from reasoning, evidence, and tool results (if available)
- Ensure the final answer directly addresses the original query
- Provide a clear and concise final answer
- For multiple choice questions, respond with EXACTLY the chosen option text
- Do not add any extra words, explanations, or formatting
- If uncertain, choose the most logically sound option based on the reasoning

Reasoning Analysis:
{full_context}

Query:
{query}

Based on the reasoning above, provide the final answer:"""
            answer = self.llm.generate(prompt, temperature=self.config.temperature, max_tokens=200)
        return answer.strip()

    def _build_medical_atomic_fact_decomposition_system_prompt(self, full_context: str) -> str:
        """Full context in this case is the reasoning steps."""
        system_prompt1 = f"""You are a medical expert in evaluating how factual a medical sentence is. You break down a sentence into as many facts as possible using step-by-step reasoning while addressing the 7 MedScoreTaxonomy issues.

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

        system_prompt2 = """You are a meticulous medical expert specializing in information extraction. Your task is to evaluate a medical sentence and decompose it into individual, verifiable facts. You must follow a rigorous step-by-step reasoning process to ensure accuracy and adherence to the 7 MedScore Taxonomy issues.

CRITICAL INSTRUCTIONS:
- Each fact must focus on a SINGLE medical concept.
- Each fact must be a simple, declarative sentence.
- Each fact must be objective and independently verifiable.
- All crucial medical details (modifiers, conditions, dosage, frequency) must be preserved.
- If a sentence contains no verifiable medical information, you must output "No verifiable claim".

REASONING CHAIN OF THOUGHT:
You must follow these 7 steps to break down the sentence. Think through each step before providing the final facts.

Step 1: IDENTIFY THE SENTENCE'S CORE PURPOSE. Is this sentence describing a personal action, a subjective experience, or is it reporting a medical fact/statement?
Step 2: FILTER UNVERIFIABLE CONTENT. If the sentence is a personal narrative (e.g., "I spoke to the doctor") or a subjective statement, it contains no verifiable claims. If it reports a medical statement (e.g., "The doctor said that..."), focus only on the content of that statement for the next steps.
Step 3: RESOLVE CONTEXT AND PRONOUNS. Identify all pronouns (it, they, this) and general terms (the medication, the condition) and replace them with the specific medical entities from the provided CONTEXT. This is critical for creating standalone facts.
Step 4: DECOMPOSE INTO ATOMIC FACTS. Break down the verifiable content into the smallest possible pieces of information. Each piece should represent a single, distinct medical concept.
Step 5: PRESERVE MEDICAL NUANCE. For each atomic fact, ensure that all relevant medical modifiers are included. This includes qualifiers (e.g., may, could, often), frequency (e.g., twice daily), duration (e.g., for 7 days), and cause-effect relationships.
Step 6: REFORMAT AS DECLARATIVE SENTENCES. Convert all facts into simple, declarative statements. (e.g., "The doctor believes that [claim]" becomes "[claim]").
Step 7: REVIEW AND REFINE. Read the final list of facts. Remove any duplicates and check that every fact is objective, verifiable, and accurately reflects the information in the original sentence.

HERE ARE SOME EXAMPLES WITH THE REQUIRED REASONING:

Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit. Considering your tetanus vaccine expired in 2020 and you've got a dirty wound from the Spartan race, your doctor recommends getting a tetanus booster vaccine as soon as possible. They also mentioned that you were due for a booster anyway since it's been more than 3 years since your last vaccine. Your doctor is a bit puzzled as to why you were given IGG shots instead of a vaccine during your last visit, but that's not a concern for now. They just want to make sure you get the booster vaccine to be on the safe side. It's best to schedule an appointment for the booster vaccine as soon as possible to avoid any potential risks.

Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
Step 1: IDENTIFY THE SENTENCE'S CORE PURPOSE. The sentence describes the speaker's personal action ("I spoke to your doctor").
Step 2: FILTER UNVERIFIABLE CONTENT. Because the sentence is a personal narrative about an action, it does not contain a verifiable medical fact. It's about who spoke to whom, not a medical statement. Therefore, the entire sentence is filtered out.
Steps 3-7: Not applicable as no verifiable content was found.

Facts:
- No verifiable claim
---
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects. The doctor mentioned that the anabolic cycle your partner is on is quite intense and requires careful monitoring for potential issues such as infertility, mood swings, and problems related to weight gain, including snoring and possible sleep apnea. They also emphasized the importance of considering the long-term effects of using these substances, particularly when they are stopped. The doctor's primary concern is that your partner's underlying condition, Addison's disease, may not significantly complicate things if well-treated, but it could become an issue when the anabolic cycle is stopped. They strongly advise that your partner consult with a medical professional, ideally their endocrinologist, to discuss the potential risks and consequences of using these substances, especially given their pre-existing condition. It's essential to have an open and honest conversation with a healthcare professional to ensure your partner's safety and well-being. I would encourage you to support your partner in seeking medical advice, and I'm happy to facilitate a discussion with their doctor if needed.

Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
Step 1: IDENTIFY THE SENTENCE'S CORE PURPOSE. The sentence's purpose is to report a medical statement made by a doctor regarding "these substances."
Step 2: FILTER UNVERIFIABLE CONTENT. The phrase "The doctor noted that" indicates a reported statement. I will focus on the content that follows: "while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects." This content is verifiable.
Step 3: RESOLVE CONTEXT AND PRONOUNS. The context mentions "anabolic steroids." Therefore, "these substances" and "they" should be replaced with "Anabolic steroids."
Step 4: DECOMPOSE INTO ATOMIC FACTS. The sentence contains four distinct concepts: 1) a positive effect on muscle health, 2) a positive effect on bone health, 3) the presence of risks, and 4) the presence of side effects.
Step 5: PRESERVE MEDICAL NUANCE. The modifier "may have" applies to the positive effects. The adjective "significant" applies to risks. The adjective "potential" applies to side effects. These must be preserved.
Step 6: REFORMAT AS DECLARATIVE SENTENCES. I will combine the resolved nouns from Step 3 with the decomposed facts from Step 4 and the nuance from Step 5 to create simple declarative sentences.
Step 7: REVIEW AND REFINE. The four resulting facts are distinct, objective, and accurately reflect the source sentence. No duplicates exist.

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids carry significant risks.
- Anabolic steroids carry potential side effects.
---
Context: The patient was diagnosed with diabetes and prescribed metformin. The doctor explained that this medication helps control blood sugar levels and that it should be taken with meals. They also mentioned that these tablets can cause gastrointestinal side effects, but those usually subside after a few weeks. The patient asked about insulin, and the doctor said that it might be needed later if this treatment doesn't work effectively.

Please breakdown the following sentence into independent facts: The doctor explained that this medication helps control blood sugar levels and that it should be taken with meals.

Reasoning:
Step 1: IDENTIFY THE SENTENCE'S CORE PURPOSE. The sentence reports a doctor's explanation about a medication.
Step 2: FILTER UNVERIFIABLE CONTENT. The framing "The doctor explained that" introduces verifiable medical information. I will extract the content: "this medication helps control blood sugar levels and that it should be taken with meals."
Step 3: RESOLVE CONTEXT AND PRONOUNS. The context states the patient was "prescribed metformin." Thus, "this medication" and "it" both refer to "metformin."
Step 4: DECOMPOSE INTO ATOMIC FACTS. The sentence has two distinct facts: 1) its effect on blood sugar, and 2) its administration instructions.
Step 5: PRESERVE MEDICAL NUANCE. There are no complex modifiers to preserve in this case. The key actions are "helps control" and "should be taken with meals."
Step 6: REFORMAT AS DECLARATIVE SENTENCES. I will create two simple sentences using the resolved noun "metformin."
Step 7: REVIEW AND REFINE. The two facts are clear, distinct, and accurate.

Facts:
- Metformin helps control blood sugar levels.
- Metformin should be taken with meals.

Now, for your task, follow the same reasoning process."""

        system_prompt3 = """You are a meticulous medical expert specializing in information extraction. Your task is to decompose a medical sentence into individual, verifiable facts by following a rigorous reasoning process designed to resolve the 7 common issues of the MedScore Taxonomy.

---
GUIDING PRINCIPLE: A Systematic Approach to Fact Extraction
Your goal is to transform complex sentences into a list of simple, standalone facts. This process is designed to systematically prevent common errors, including:
1.  Extracting unverifiable personal narratives.
2.  Losing critical medical nuance (e.g., modifiers, dosage).
3.  Creating claims that depend on outside context (e.g., using pronouns).
4.  Generating complex claims with multiple concepts.
5.  Failing to convert questions or commands into declarative statements.
6.  Introducing information not present in the original text (hallucinations).
7.  Producing redundant or duplicate facts.
---

CRITICAL INSTRUCTIONS:
- Each fact must focus on a SINGLE medical concept.
- Each fact must be a simple, declarative sentence.
- All crucial medical details (modifiers, conditions, dosage) must be preserved.
- If a sentence contains no verifiable medical information, you MUST output "No verifiable claim".

REASONING CHAIN OF THOUGHT:
You must follow these 6 steps in order. Each step solves specific MedScore issues.

Step 1: TRIAGE - Is this sentence a Narrative or a Factual Report?
First, analyze the sentence's primary function. Is it describing a personal interaction, conversation, or intention (e.g., "I spoke with...", "We discussed...")? If so, it is an UNVERIFIABLE NARRATIVE. The process stops here. Output "No verifiable claim". If the sentence is reporting a medical statement (e.g., "The doctor said that..."), proceed to the next step.
(This step solves: Unverifiable Claims)

Step 2: ISOLATE & GROUND - What is the core verifiable information?
Isolate the exact clause containing the medical information. This grounds the extraction process, ensuring no outside information or HALLUCINATIONS are introduced.
(This step solves: Hallucinated Claims)

Step 3: DECONTEXTUALIZE - Make the claim standalone.
Replace ALL pronouns (it, they, this) and general terms (the medication, the condition) with the specific medical entities from the provided CONTEXT. This is critical to resolve CONTEXT-DEPENDENT CLAIMS.
(This step solves: Context-Dependent Claims)

Step 4: DECOMPOSE - Break it down into atomic facts.
Break the verifiable clause into the smallest possible pieces of information. Each piece must represent a single, distinct medical concept. This prevents COMPLEX/NESTED CLAIMS.
(This step solves: Complex/Nested Claims)

Step 5: RECONSTRUCT - Build complete, declarative facts.
For each atomic fact from Step 4, reconstruct a full sentence. Ensure you preserve all original medical nuance (modifiers like 'may', 'potential'; frequency like 'twice daily'). Convert any questions or commands into a DECLARATIVE FORMAT.
(This step solves: Incomplete Claims, Imperative/Interrogative Format)

Step 6: REVIEW & DEDUPLICATE - Final quality check.
Read the final list of facts. Remove any REDUNDANT claims or minor rephrasings of the same fact. Ensure every fact is objective, verifiable, and accurately reflects the information isolated in Step 2.
(This step solves: Redundant Claims)

HERE ARE SOME EXAMPLES WITH THE REQUIRED REASONING:

Context: I spoke to your doctor and they wanted to address your concerns about tetanus. Since you've had your primary tetanus shots as a child, you don't need immunoglobulin (IGG) shots, and they were actually unnecessary during your last visit.
Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
Step 1: TRIAGE. The sentence "I spoke to your doctor and they wanted to address..." describes a personal interaction and an intention. This is a classic unverifiable narrative. The process stops here.

Facts:
- No verifiable claim
---
Context: I spoke to your doctor... The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.

Reasoning:
Step 1: TRIAGE. The frame "The doctor noted that..." indicates a factual report. I will proceed.
Step 2: ISOLATE & GROUND. The core verifiable information is "while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Step 3: DECONTEXTUALIZE. From the context, "these substances" and "they" refer to "Anabolic steroids."
Step 4: DECOMPOSE. This clause contains four atomic concepts: effect on muscle, effect on bone, presence of risks, and presence of side effects.
Step 5: RECONSTRUCT. I will build four declarative sentences, making sure to preserve the nuance: "may have" for positive effects, "significant" for risks, and "potential" for side effects.
Step 6: REVIEW & DEDUPLICATE. The four resulting facts are distinct and non-redundant.

Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids carry significant risks.
- Anabolic steroids carry potential side effects.

Now, for your task, follow the same reasoning process."""

        system_prompt4 = """You are a meticulous medical expert specializing in information extraction. Your task is to decompose a medical sentence into individual, verifiable facts by following a rigorous reasoning process designed to resolve the 7 common issues of the MedScore Taxonomy.

---
GUIDING PRINCIPLE: A Systematic Approach to Fact Extraction
Your goal is to transform complex sentences into a list of simple, standalone facts. This process is designed to systematically prevent common errors, including:
1.  Extracting unverifiable personal narratives.
2.  Losing critical medical nuance (e.g., modifiers, dosage).
3.  Creating claims that depend on outside context (e.g., using pronouns).
4.  Generating complex claims with multiple concepts.
5.  Failing to convert questions or commands into declarative statements.
6.  Introducing information not present in the original text (hallucinations).
7.  Producing redundant or duplicate facts.
---

CRITICAL INSTRUCTIONS:
- Each fact must focus on a SINGLE medical concept.
- Each fact must be a simple, declarative sentence.
- All crucial medical details (modifiers, conditions, dosage) must be preserved.
- If a sentence contains no verifiable medical information, you MUST output "No verifiable claim".

REASONING CHAIN OF THOUGHT:
You must follow these 6 steps in order. Each step solves specific MedScore issues.

Step 1: TRIAGE - Is this sentence a Narrative or a Factual Report?
First, analyze the sentence's primary function. Is it describing a personal interaction (e.g., "I spoke with...")? If so, it is an UNVERIFIABLE NARRATIVE. The process stops here. If the sentence is reporting a medical statement (e.g., "The doctor said that..."), proceed.
(This step solves: Unverifiable Claims)

Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME.
Separate the reporting frame (e.g., "The doctor believes that...", "The study shows that...") from the core medical content. The frame itself is not a verifiable fact. Your focus for the next steps is ONLY the medical content clause.
(This step solves: Hallucinated Claims by grounding in the specific content)

Step 3: DECONTEXTUALIZE - Make the claim standalone.
Replace ALL pronouns (it, they, your) and general terms (the medication) with the specific entities from the context. Generalize patient-specific terms like "your symptoms" to "symptoms" to create a universal fact.
(This step solves: Context-Dependent Claims)

Step 4: DECOMPOSE - Break it down into atomic facts.
Break the verifiable clause into the smallest possible pieces of information. Each piece must represent a single, distinct medical concept.
- Handle Conjunctions: If a subject is linked to multiple medical concepts (e.g., 'symptoms are related to A and B'), you must create a separate fact for each link (Fact 1: 'symptoms are related to A', Fact 2: 'symptoms are related to B').
(This step solves: Complex/Nested Claims)

Step 5: RECONSTRUCT - Build complete, declarative facts.
For each atomic fact, reconstruct a full sentence. Ensure you preserve all original medical nuance (modifiers like 'may', 'could'; frequency like 'twice daily'). Convert any questions or commands into a DECLARATIVE FORMAT.
(This step solves: Incomplete Claims, Imperative/Interrogative Format)

Step 6: REVIEW & DEDUPLICATE - Final quality check.
Read the final list of facts. Remove any REDUNDANT claims. Ensure every fact is objective, verifiable, and accurately reflects the information isolated in Step 2.
(This step solves: Redundant Claims)

HERE ARE SOME EXAMPLES WITH THE REQUIRED REASONING:

Context: (Full context about tetanus)
Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
Step 1: TRIAGE. The sentence "I spoke to your doctor..." describes a personal interaction. This is an unverifiable narrative. The process stops here.

Facts:
- No verifiable claim
---
Context: (Full context about anovulatory cycles and dysmenorrhea)
Please breakdown the following sentence into independent facts: They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.

Reasoning:
Step 1: TRIAGE. The frame "They believe that..." indicates a factual report. I will proceed.
Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is "They believe that". The core medical content is: "your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."
Step 3: DECONTEXTUALIZE. "your symptoms" is generalized to "symptoms".
Step 4: DECOMPOSE. I will break the content into all its distinct concepts, applying the conjunction rule:
 - Concept 1: The link between symptoms and the first condition.
 - Concept 2: The definition of the first condition.
 - Concept 3: The link between symptoms and the second condition (This is the crucial "and" rule).
 - Concept 4: The definition of the second condition.
Step 5: RECONSTRUCT. I will build four declarative sentences, preserving the modifier "could be related to" for the linkage facts.
 - Fact 1: Symptoms could be related to anovulatory cycles.
 - Fact 2: Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
 - Fact 3: Symptoms could be related to primary dysmenorrhea.
 - Fact 4: Primary dysmenorrhea is a condition that causes painful periods.
Step 6: REVIEW & DEDUPLICATE. The four facts are distinct, non-redundant, and accurately reflect the isolated medical content.

Facts:
- Symptoms could be related to anovulatory cycles.
- Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
- Symptoms could be related to primary dysmenorrhea.
- Primary dysmenorrhea is a condition that causes painful periods.

Now, for your task, follow the same reasoning process."""

        system_prompt = """You are a meticulous medical expert specializing in information extraction. Your task is to decompose a medical sentence into individual, verifiable facts by following a rigorous reasoning process designed to resolve the 7 common issues of the MedScore Taxonomy.

---
GUIDING PRINCIPLE: A Systematic Approach to Atomic Fact Extraction
Your goal is to transform complex sentences into a list of simple, standalone facts. This process is designed to systematically prevent common errors, including:
1.  Extracting unverifiable personal narratives.
2.  Losing critical medical nuance (e.g., modifiers, dosage).
3.  Creating claims that depend on outside context (e.g., using pronouns).
4.  Generating complex claims with multiple concepts.
5.  Failing to convert questions or commands into declarative statements.
6.  Introducing information not present in the original text (hallucinations).
7.  Producing redundant or omitted claims.
---

CRITICAL INSTRUCTIONS:
- Each fact must focus on a SINGLE medical concept.
- Each fact must be a simple, declarative sentence.
- All crucial medical details (modifiers, conditions, dosage) must be preserved.
- If a sentence contains no verifiable medical information, you MUST output "No verifiable claim".

REASONING CHAIN OF THOUGHT:
You must follow these 6 steps in order. Each step solves specific MedScore issues.

Step 1: TRIAGE - Is this sentence a Narrative or a Factual Report?
First, analyze the sentence's primary function. Is it describing a personal interaction (e.g., "I spoke with...")? If so, it is an UNVERIFIABLE NARRATIVE. The process stops here. If the sentence is reporting a medical statement (e.g., "The doctor said that..."), proceed.
(This step solves: Unverifiable Claims)

Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME.
Separate the reporting frame (e.g., "The doctor believes that...", "The study shows that...") from the core medical content. The frame itself is not a verifiable fact. Your focus for the next steps is ONLY the medical content clause.
(This step solves: Incorrectly structured claims)

Step 3: DECONTEXTUALIZE - Make the claim standalone.
Replace ALL pronouns (it, they, your) and general terms (the medication) with the specific entities from the context. Generalize patient-specific terms like "your symptoms" to "what is your symptoms in the context" to create a universal fact.
(This step solves: Context-Dependent Claims)

Step 4: DECOMPOSE - Break it down into atomic facts.
Break the verifiable clause into the smallest possible pieces of information. Each piece must represent a single, distinct medical concept.
 - Handle Conjunctions: If a subject is linked to multiple medical concepts (e.g., 'symptoms are related to A and B'), you must create a separate fact for each link (Fact 1: 'symptoms are related to A', Fact 2: 'symptoms are related to B').
(This step solves: Hallucinated Claims)

Step 5: RECONSTRUCT - Build complete, declarative facts.
For each atomic fact, reconstruct a full sentence. Ensure you preserve all original medical nuance (modifiers like 'may', 'could'; frequency like 'twice daily'). Convert any questions or commands into a DECLARATIVE FORMAT.
(This step solves: Incomplete Claims, Incorrectly structured)

Step 6: REVIEW - Final Quality and Coverage Check.
Read the final list of facts and perform two critical checks:
 - Deduplication Check: Remove any REDUNDANT claims or minor rephrasings of the same fact. A claim is redundant if it is a direct duplicate OR if it is a composite of other, more atomic claims.
 - Coverage Check: Compare your final facts against the isolated content from Step 2. Have all important medical concepts been extracted? This prevents OMITTED claims.
(This step solves: Redundant Claims, Omitted Claims)

HERE ARE SOME EXAMPLES WITH THE REQUIRED REASONING:

Context: (Full context about tetanus)
Please breakdown the following sentence into independent facts: I spoke to your doctor and they wanted to address your concerns about tetanus.

Reasoning:
Step 1: TRIAGE. The sentence "I spoke to your doctor..." describes a personal interaction. This is an unverifiable narrative. The process stops here.

Facts:
- No verifiable claim
---
Context: I spoke to your doctor, and they expressed concerns about the safety of using anabolic steroids, particularly in combination with the medications your partner is already taking for Addison's disease. The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Please breakdown the following sentence into independent facts: The doctor noted that while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects.
Reasoning:
Step 1: TRIAGE. The frame "The doctor noted that..." indicates a factual report. I will proceed.
Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is "The doctor noted that". The core medical content is: "while these substances may have positive effects on muscle and bone health, they also carry significant risks and potential side effects."
Step 3: DECONTEXTUALIZE. "these substances" and "they" are replaced with "Anabolic steroids" from the context.
Step 4: DECOMPOSE. I will break the content into four distinct concepts:
 - Concept 1: Positive effect on muscle health.
 - Concept 2: Positive effect on bone health.
 - Concept 3: The presence of significant risks.
 - Concept 4: The presence of potential side effects.
Step 5: RECONSTRUCT. I will build four declarative sentences, preserving the original modifiers.
 - Fact 1: Anabolic steroids may have positive effects on muscle health.
 - Fact 2: Anabolic steroids carry significant risks and potential side effects.
 - Fact 3: Anabolic steroids carry significant risks.
 - Fact 4: Anabolic steroids carry potential side effects.
Step 6: REVIEW - Final Quality and Coverage Check.
 - Deduplication Check: I have identified that Fact 2 ("...carry significant risks and potential side effects") is a composite of Fact 3 and Fact 4. It is redundant because it's not atomic. I will remove Fact 2 and keep the more specific facts (3 and 4).
 - Coverage Check: I am comparing my list to the concepts from Step 4. I have omitted Concept 2 ("positive effects on bone health"). I must add this fact to the final list to ensure complete coverage.
Facts:
- Anabolic steroids may have positive effects on muscle health.
- Anabolic steroids may have positive effects on bone health.
- Anabolic steroids carry significant risks.
- Anabolic steroids carry potential side effects.
---
Context: I spoke to your doctor and they wanted to address your concerns about your irregular periods and extreme pain. They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.
Please breakdown the following sentence into independent facts: They believe that your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods.

Reasoning:
Step 1: TRIAGE. The frame "They believe that..." indicates a factual report. I will proceed.
Step 2: ISOLATE CONTENT & STRIP REPORTING FRAME. The reporting frame is "They believe that". The core medical content is: "your symptoms could be related to anovulatory cycles, which means that your body is not releasing an egg during your menstrual cycle, and primary dysmenorrhea, which is a condition that causes painful periods."
	
Step 3: DECONTEXTUALIZE. "your symptoms" is generalized to "Irregular periods and extreme pain", "your body" is generalized to "the body", "your menstrual cycle" is generalized to "the menstrual cycle".
Step 4: DECOMPOSE. I will break the content into all its distinct concepts, applying the conjunction rule:
 - Concept 1: The link between symptoms and the first condition.
 - Concept 2: The definition of the first condition.
 - Concept 3: The link between symptoms and the second condition (This is the crucial "and" rule).
 - Concept 4: The definition of the second condition.
Step 5: RECONSTRUCT. I will build four declarative sentences, preserving the modifier "could be related to" for the linkage facts.
 - Fact 1: Irregular periods and extreme pain could be related to anovulatory cycles.
 - Fact 2: Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
 - Fact 3: Irregular periods and extreme pain could be related to primary dysmenorrhea.
 - Fact 4: Primary dysmenorrhea is a condition that causes painful periods.
Step 6: REVIEW - Final Quality and Coverage Check.
 - Deduplication Check: The four facts are distinct and non-redundant.
 - Coverage Check: I've compared the facts to the content from Step 2. All key concepts (the two conditions, their definitions, and their link to symptoms) have been extracted. No important information was omitted.

Facts:
- Irregular periods and extreme pain could be related to anovulatory cycles.
- Anovulatory cycles mean that the body is not releasing an egg during the menstrual cycle.
- Irregular periods and extreme pain could be related to primary dysmenorrhea.
- Primary dysmenorrhea is a condition that causes painful periods.

OUTPUT FORMAT:
Reasoning:
Step 1: [Step 1 reasoning]
Step 2: [Step 2 reasoning]
...
Facts:
- [Fact 1]
- [Fact 2]
...

Now, for your task, follow the same reasoning process."""
        return system_prompt


# Convenience factory functions
def create_pipeline(
        llm: LLMProvider,
        auto_config: bool = False,
        reasoning_strategy: str = "cot",
        use_rag: bool = False,
        documents_path: str = "",
        enable_tools: bool = False,
        max_tool_steps: int = 10,
        enable_atomic_fact_decomposition: bool = False,
        temperature: float = 0.7,
        verbose: bool = False
) -> UnifiedPipeline:
    config = PipelineConfig(
        auto_config=auto_config,
        reasoning_strategy=reasoning_strategy,
        use_rag=use_rag,
        documents_path=documents_path,
        enable_tools=enable_tools,
        max_tool_steps=max_tool_steps,
        enable_atomic_fact_decomposition=enable_atomic_fact_decomposition,
        temperature=temperature,
        verbose=verbose
    )
    return UnifiedPipeline(llm, config)


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
