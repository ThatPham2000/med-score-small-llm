import json

import jsonlines

input_file = '/Users/that.phamvan/my_ws/master/med-score-vi/data/hospital_108/page1_augmented.jsonl'

# Load data
with jsonlines.open(input_file) as reader:
    dataset = {item['id']: item['doctor_response'] for item in reader.iter()}

# Save to JSON file
output_file = '/Users/that.phamvan/my_ws/master/med-score-vi/data/hospital_108_evidence.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
