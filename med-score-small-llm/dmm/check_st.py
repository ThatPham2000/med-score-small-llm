import jsonlines
import ollama

import json


def extract_parse_failed_claims(input_filepath, output_filepath):
    extracted_count = 0

    # Mở file đọc và file ghi cùng lúc
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
            open(output_filepath, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Bỏ qua các dòng trống
            if not line.strip():
                continue

            try:
                # Chuyển đổi chuỗi JSON thành dictionary
                data = json.loads(line)

                # Kiểm tra điều kiện lọc
                if data.get("claim_quality_type") == "Parse Failed":
                    # Ghi dòng thỏa mãn điều kiện vào file mới
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    extracted_count += 1

            except json.JSONDecodeError as e:
                print(f"Bỏ qua dòng lỗi định dạng JSON: {e}")

    print(f"Đã trích xuất thành công {extracted_count} dòng và lưu vào '{output_filepath}'.")


def extract_claims(input_filepath, output_filepath, label):
    extracted_count = 0

    # Mở file đọc và file ghi cùng lúc
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
            open(output_filepath, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Bỏ qua các dòng trống
            if not line.strip():
                continue

            try:
                # Chuyển đổi chuỗi JSON thành dictionary
                data = json.loads(line)

                # Kiểm tra điều kiện lọc
                if data.get("claim_quality_type") == label:
                    # Ghi dòng thỏa mãn điều kiện vào file mới
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    extracted_count += 1

            except json.JSONDecodeError as e:
                print(f"Bỏ qua dòng lỗi định dạng JSON: {e}")

    print(f"Đã trích xuất thành công {extracted_count} dòng và lưu vào '{output_filepath}'.")

''' total: 1204,30
1. Unverifiable: Personal narratives or empathy. 0,0
2. Incorrectly structured: Questions, commands, or contains reporting frames. 7 (5.8%),1
3. Context-dependent: Contains unresolved pronouns or vague terms. 28 (2.3%), 1
4. Incomplete: Missing critical modifiers (may, severe) or conditions (if, when). 46 (3.8%), 1
5. Hallucinated: Adds info not in "Atomic Claim" or distorts meaning. 128 (10.63%),3
6. Redundant: A duplicate or a composite of other claims. 121 (10.05%),3
7. Valid: A perfect, standalone atomic fact. 874 (72.6%),21
'''
if __name__ == '__main__':
    extract_claims(
        "/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/small_llm/ministral3_14b/small_llm_provided_claim_quality_evaluations.jsonl",
        "/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/dmm/small_llm_ministral3_AskDocsAI_claim_quality_Redundant.jsonl",
        "Redundant"
    )

import json
import os


def remove_parse_failed_claims_inplace(filepath):
    # Tạo đường dẫn cho file tạm
    temp_filepath = filepath + ".tmp"
    removed_count = 0

    try:
        # Mở file gốc để đọc và file tạm để ghi
        with open(filepath, 'r', encoding='utf-8') as infile, \
                open(temp_filepath, 'w', encoding='utf-8') as outfile:

            for line in infile:
                # Bỏ qua dòng trống
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Nếu thỏa mãn điều kiện "Parse Failed", ta bỏ qua (không ghi vào file tạm)
                    if data.get("claim_quality_type") == "Parse Failed":
                        removed_count += 1
                        continue

                        # Ghi các dòng KHÔNG THỎA MÃN điều kiện vào file tạm
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')

                except json.JSONDecodeError:
                    # Nếu dòng bị lỗi định dạng JSON, giữ nguyên nó lại để tránh mất dữ liệu gốc
                    outfile.write(line)

        # Sau khi ghi xong file tạm thành công, tiến hành ghi đè file gốc
        os.replace(temp_filepath, filepath)
        print(f"Xử lý hoàn tất! Đã xóa {removed_count} dòng 'Parse Failed' trực tiếp từ file '{filepath}'.")

    except Exception as e:
        # Nếu có lỗi xảy ra trong quá trình xử lý, xóa file tạm đi để dọn dẹp hệ thống
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        print(f"Đã xảy ra lỗi: {e}. File gốc chưa bị thay đổi.")

if __name__ == '__main2__':
    with jsonlines.open('/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/messages.jsonl', 'r') as reader:
        messages = [item for item in reader.iter()]

    # from openai import OpenAI
    #
    # client = OpenAI(
    #     base_url="https://api.together.ai/v1",
    #     api_key="tgp_v1_igV2ogsxptxMLff74BLpJtAq3FZkExRysVWh0z2t3wU",
    # )
    #
    # result = []
    # for message in messages:
    #     print(message)
    #     completion = client.chat.completions.create(
    #         model="openai/gpt-oss-120b",
    #         messages=message,
    #
    #     )
    #     result.extend(completion.choices[0].message)

    result = []

    for message in messages:
        print(message)
        completion = ollama.Client().chat(
            model='gpt-oss:120b-cloud',
            messages=message,
            options={
                "seed": 42,
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 10240,
            }
        )
        result.append(completion.message.content)

    # save to local file

    with jsonlines.open('/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/dmm/message_result.jsonl', 'w') as writer:
        for item in result:
            writer.write(item)