from typing import List
from hashlib import md5


def vectorize(text: str, num_features: int, max_length: int) -> List[int]:
    splitted = text.split(sep=" ")
    vectorized = [int(md5(w.encode('utf-8')).hexdigest(), 16) % num_features for w in splitted][:max_length]
    vectorized += [0] * (max_length - len(vectorized))
    return vectorized
