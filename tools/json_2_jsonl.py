import json

from jsonlines import jsonlines

input_file = '//data/hospital_108/page1.json'
output_file = '//data/hospital_108/page1.jsonl'

# Load JSON data
with open(input_file, 'r') as reader:
    json_data = json.load(reader)

# Convert to JSONL format
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(json_data)
