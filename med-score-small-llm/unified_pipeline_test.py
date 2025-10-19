from unified_pipeline import create_llm_provider, create_pipeline


def main():
    """Test the optimized pipeline"""

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