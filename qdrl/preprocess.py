from typing import List
from hashlib import md5


def vectorize(text: str, num_features: int, max_size: int) -> List[int]:
    splitted = text.split(sep=" ")
    vectorized = [int(md5(w.encode('utf-8')).hexdigest(), 16) % num_features for w in splitted][:max_size]
    vectorized += [0] * (max_size - len(vectorized))
    return vectorized
