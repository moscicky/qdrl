import re
import unicodedata
from typing import List
from hashlib import md5


def vectorize_word(text: str, num_features: int, max_length: int) -> List[int]:
    splitted = text.split(sep=" ")
    buckets = num_features - 1
    vectorized = [(int(md5(w.encode('utf-8')).hexdigest(), 16) % buckets) + 1 for w in splitted][:max_length]
    vectorized += [0] * (max_length - len(vectorized))
    return vectorized


clean_pattern = re.compile(r"[^a-z0-9 ]")


def clean_phrase(text: str) -> str:
    decoded = unicodedata.normalize("NFKD", text
                                    .replace('ł', 'l')
                                    .replace('Ł', 'L')
                                    )

    return clean_pattern.sub("", decoded.lower()).strip()
