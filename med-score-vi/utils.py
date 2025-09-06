from itertools import islice
from typing import List, Dict, Any, Iterable

import spacy

# nlp = spacy.load('vi_spacy_model')
nlp = spacy.load("en_core_web_sm")


# input: passage = "Hello world. This is an example."
# output: [{'text': 'Hello world.', 'span_start': 0, 'span_end': 12}, {'text': 'This is an example.', 'span_start': 13, 'span_end': 32}]
def parse_sentences(
        passage: str,
) -> List[Dict[str, Any]]:
    doc = nlp(passage)
    sentences = []
    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "span_start": sent.start_char,
            "span_end": sent.end_char
        })
    return sentences


# input: iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#        n = 3
# output: [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20)]
def chunker(
        iterable: Iterable,
        n: int
):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


# input: claims = ["- Claim 1", "- Claim 2", "No verifiable claim"]
# output: ["Claim 1", "Claim 2"]
def process_claim(claims: List[str]) -> List[str]:
    # drop - in front of the claim
    claims = [claim.strip('-').strip() for claim in claims]

    # remove 'no verifiable claim' from the claims
    claims = [claim for claim in claims if 'no verifiable claim' not in claim.lower()]

    return claims
