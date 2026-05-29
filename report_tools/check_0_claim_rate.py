# if __name__ == '__main__':
#     import jsonlines
#
#     with jsonlines.open('/Users/that.phamvan/my_ws/master/med-score-small-llm/med-score-small-llm/dmm_medscore_phi4_verified/medscore_provided_verifications.jsonl', 'r') as reader:
#         verifications = [item for item in reader.iter()]
#     print(f"len decompositions: {len(verifications)}")
#     claim_count = 0
#     for item in verifications:
#         claim_count += len(item['claims'])
#     print(f"claim count: {claim_count}")