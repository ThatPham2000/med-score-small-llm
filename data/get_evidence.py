import json


def process_data(input_file, output_file):
    result_dict = {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                record_id = data.get("id")
                doctor_response = data.get("doctor_response")

                if record_id:
                    result_dict[record_id] = doctor_response

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

        print(f"Saved at: {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found {input_file}.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the input file.")


if __name__ == '__main__':
    input_filepath = '/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_MedLFQA/live_qa_test_MedLFQA_medscore.jsonl'
    output_filepath = '/Users/that.phamvan/my_ws/master/med-score-small-llm/data/dataset_MedLFQA/live_qa_test_MedLFQA_medscore_evidence.json'

    process_data(input_filepath, output_filepath)
