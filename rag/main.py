import uuid
from typing import List, Dict

import pandas as pd
from jsonlines import jsonlines


def download_rag_dataset():
    splits = {'train': 'data/train-00000-of-00001-f0c158413defd454.parquet',
              'test': 'data/test-00000-of-00001-06d83c58a8ea10e8.parquet'}
    df = pd.read_parquet("hf://datasets/neural-bridge/rag-dataset-1200/" + splits["test"])

    # Add UUID column as first column
    df.insert(0, 'uuid', [str(uuid.uuid4()) for _ in range(len(df))])

    print(df.head())
    print(f"Total rows: {len(df)}")

    # save to jsonl
    df.to_json("rag_test.jsonl", orient="records", lines=True)
    print("Saved to rag_test.jsonl")


def read_rag_rows() -> List[Dict[str, str]]:
    with jsonlines.open(
            '/Users/that.phamvan/my_ws/master/med-score-small-llm/rag/rag_test.jsonl',
            'r') as reader:
        rows = [item for item in reader.iter()]
    return rows


if __name__ == '__main__':
    print('main')
    download_rag_dataset()
