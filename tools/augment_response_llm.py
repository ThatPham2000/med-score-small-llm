import asyncio
import json
from itertools import islice
from typing import List, Dict, Iterable

import nest_asyncio
import ollama
from jsonlines import jsonlines
from tqdm import tqdm

nest_asyncio.apply()

input_file = '/Users/that.phamvan/my_ws/master/med-score-vi/data/hospital_108/page1.json'

# Load JSON data
with open(input_file, 'r') as reader:
    json_data = json.load(reader)

messages = []
for item in json_data:
    messages.append([
        {"role": "system", "content": "Bạn là một trợ lý y tế AI, với nhiệm vụ cung cấp thông tin y khoa chính xác và dễ hiểu. Quan trọng nhất, hãy luôn bắt đầu câu trả lời bằng một lời chia sẻ, thể hiện sự đồng cảm sâu sắc với tình trạng hoặc câu hỏi của người dùng trước khi trình bày thông tin chuyên môn. Giọng văn của bạn phải thật ấm áp, quan tâm và đáng tin cậy. Câu trả lời phải là bằng tiếng Việt, định dạng văn bản thuần."},
        {
            "role": "user",
            "content": f"{item['doctor_response']}.\n\nHãy thêm sự đồng cảm vào đoạn văn này, làm cho bệnh nhân cảm thấy được thấu hiểu và an ủi, mà không thêm bất kỳ thông tin y khoa nào. Trình bày lại đoạn văn theo cách của bạn mà không thay đổi thông tin."
        },
    ])

# messages=[messages[0]]
# json_data=[json_data[0]]

async def batch_response(batch: List[List[Dict[str, str]]]) -> List[str]:
    async_responses = [
        ollama.AsyncClient().chat(
            model='gpt-oss:20b',
            messages=x,
            options={
                "temperature": 0.3,
                "top_p": 0.1
            }
        )
        for x in batch
    ]
    return await asyncio.gather(*async_responses)


def chunker(iterable: Iterable, n: int):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


all_completions = []
batch_size = 32
n_iter = len(messages) // batch_size
for batch in tqdm(chunker(messages, batch_size), desc="Augment response process", total=n_iter, ncols=0):
    completions = asyncio.run(batch_response(batch))
    all_completions.extend(completions)

result = []
for item, completion in zip(json_data, all_completions):
    item["response"] = completion.message.content
    result.append(item)

# Save to JSONL file
output_file = '/Users/that.phamvan/my_ws/master/med-score-vi/data/hospital_108/page1_augmented.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(result)
