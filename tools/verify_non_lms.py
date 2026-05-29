import string

import jsonlines
import ollama


def parse_verification_output(completion_message: str) -> float:
    generated_answer = completion_message.strip().lower()
    is_supported = 0.0

    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = 1.0
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = 0.0
        else:
            # Below logic is wrong: it gets the first occurrence of "true" and "false", and if the first true is later than the first false, it is considered true.
            # is_supported = generated_answer.index("true") > generated_answer.index("false")

            # If the last occurrence of 'true' appears later than 'false' in the output, then think the conclusion is true.
            is_supported = generated_answer.rindex("true") > generated_answer.rindex("false")
            is_supported = 1.0 if is_supported else 0.0
    else:
        generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
        is_supported = all(
            [keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
        is_supported = 1.0 if is_supported else 0.0
    return is_supported

def verify():
    evidence = '''Well the important question is here is why don't you eat sugar? And is it only about treats and sweets, or do you also avoid other food types? Carbs? Bread and rice? And what else do you eat besides vegetables? Do you eat meat and other sources of protein? Dairy products? You say you eat a lot, but a meal and a half doesn't sound like a lot on a daily basis, even if they're big meals.'''
    claim ='''Reviewing the notes from'''
    formatted_input = f"""Answer the question based on the given context.\n\n{evidence}\n\nInput: {claim} True or False?\nOutput:"""
    print(formatted_input)
    result = ollama.chat(model='gpt-oss:20b', messages=[
        {"role": "user", "content": formatted_input},
    ],options={
                    "seed": 42,
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "num_predict": 256,
                })
    completion = result.message.content
    print(f"Result: \n{result.message.content}")

    raw_output = completion.strip()
    is_supported = parse_verification_output(raw_output)
    # output = {k: v for k, v in verifier_input.items()}
    # output["raw"] = raw_output
    # output["score"] = is_supported
    print(f"Raw output: {raw_output}")
    print(f"Is supported: {is_supported}")

def report():
    with jsonlines.open('/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/medscore/ministral3_14b/med_lfqa/medscore_provided_decompositions.jsonl', 'r') as reader:
        decompositions = [item for item in reader.iter()]
    print(f"len decompositions: {len(decompositions)}")

    with jsonlines.open('/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/medscore/ministral3_14b/med_lfqa/medscore_provided_verifications.jsonl', 'r') as reader:
        verifications = [item for item in reader.iter()]
    print(f"len decompositions: {len(decompositions)}")

    # Combine results
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
        try:
            combined_output[verification['id']]['claims'].append(claim_info)
        except KeyError:
            print(f"Warning: No matching decomposition found for verification with id {verification['id']}")

    # Aggregate scores
    for idx in combined_output:
        claim_scores = [claim['score'] for claim in combined_output[idx]['claims']]
        if len(claim_scores) == 0:
            combined_output[idx]["score"] = None
        else:
            combined_output[idx]["score"] = sum(claim_scores) / len(claim_scores)

    combined_output = [v for k, v in combined_output.items()]
    with jsonlines.open('dmm_output_file.jsonl', 'w') as writer:
        writer.write_all(combined_output)

    # Calculate and display final metrics
    scores = [item['score'] for item in combined_output if item['score'] is not None]
    final_score = sum(scores) / len(scores) if scores else None

    print(f"Total responses evaluated: {len(combined_output)}")
    print(f"Responses with valid scores: {len(scores)}")
    print(f"Final MedScore: {final_score:.4f}")

    # Additional metrics for comparison
    if scores:
        import statistics

        print(f"Score statistics:")
        print(f"  Mean: {statistics.mean(scores):.4f}")
        print(f"  Median: {statistics.median(scores):.4f}")
        if len(scores) > 1:
            print(f"  Std Dev: {statistics.stdev(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
    exit(0)

if __name__ == '__main__':
    # verify()
    report()
