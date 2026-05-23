import json
from sklearn.metrics import cohen_kappa_score


def load_predictions(file_path):
    """
    Đọc file JSONL và trả về một dictionary.
    Key: id_sentenceId_claimId (để đảm bảo map đúng claim giữa 2 models)
    Value: claim_quality_type
    """
    predictions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # Tạo unique key từ các ID để map chính xác
            doc_id = data.get('id')
            sent_id = data.get('sentence_id')
            claim_id = data.get('claim_id')
            unique_key = f"{doc_id}_{sent_id}_{claim_id}"

            predictions[unique_key] = data.get('claim_quality_type')

    return predictions


def evaluate_model_agreement(file1_path, file2_path):
    """So sánh độ đồng thuận giữa 2 file kết quả."""
    model1_preds = load_predictions(file1_path)
    model2_preds = load_predictions(file2_path)

    # Tìm các claim tồn tại chung trong cả 2 file
    common_keys = set(model1_preds.keys()).intersection(set(model2_preds.keys()))

    if not common_keys:
        return "Không tìm thấy claim nào chung giữa 2 files để so sánh."

    labels_model1 = []
    labels_model2 = []
    exact_matches = 0

    for key in common_keys:
        l1 = model1_preds[key]
        l2 = model2_preds[key]

        labels_model1.append(l1)
        labels_model2.append(l2)

        if l1 == l2:
            exact_matches += 1
            print(f"{key}: {l1} vs. {l2}")


    total_common = len(common_keys)
    percentage_agreement = (exact_matches / total_common) * 100

    # Tính Cohen's Kappa
    kappa_score = cohen_kappa_score(labels_model1, labels_model2)

    # In báo cáo
    print("=" * 40)
    print("BÁO CÁO ĐỘ ĐỒNG THUẬN (AGREEMENT REPORT)")
    print("=" * 40)
    print(f"Tổng số claims đối chiếu: {total_common}")
    print(f"Số claims khớp nhau hoàn toàn: {exact_matches}")
    print(f"Tỷ lệ đồng thuận (Percentage): {percentage_agreement:.2f}%")
    print(f"Điểm Cohen's Kappa: {kappa_score:.4f}")
    print("=" * 40)

    # Phân loại độ tin cậy của Kappa
    if kappa_score < 0:
        interpretation = "Kém (Poor)"
    elif kappa_score <= 0.20:
        interpretation = "Rất yếu (Slight)"
    elif kappa_score <= 0.40:
        interpretation = "Yếu (Fair)"
    elif kappa_score <= 0.60:
        interpretation = "Trung bình (Moderate)"
    elif kappa_score <= 0.80:
        interpretation = "Tốt (Substantial)"
    else:
        interpretation = "Rất tốt (Almost Perfect)"

    print(f"-> Đánh giá mức độ đồng thuận: {interpretation}")

# Hướng dẫn sử dụng
# evaluate_model_agreement('model_1_output.jsonl', 'model_2_output.jsonl')

if __name__ == '__main__':
    file1 ="/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/dmm_small_llm_ministral3_askdocai_gpt5_classification/small_llm_provided_claim_quality_evaluations.jsonl"
    file2 ="/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/small_llm/ministral3_14b/small_llm_provided_claim_quality_evaluations.jsonl"
    evaluate_model_agreement(file1, file2)