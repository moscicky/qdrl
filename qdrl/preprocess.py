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


def vectorize_character_trigrams(text: str, num_features: int, max_length: int, offset: int = 0) -> List[int]:
    trigrams = re.findall(r'(?=(...))', text)
    buckets = num_features - 1
    vectorized = [(int(md5(t.encode('utf-8')).hexdigest(), 16) % buckets) + offset + 1 for t in trigrams][:max_length]
    vectorized += [0] * (max_length - len(vectorized))
    return vectorized


# 50k, 20k
def vectorize_combined(text: str,
                       word_unigram_num_features: int,
                       word_unigram_max_length: int,
                       char_trigrams_num_features: int,
                       char_trigrams_max_length: int) -> List[int]:
    word_unigram_vector = vectorize_word(text,
                                         num_features=word_unigram_num_features,
                                         max_length=word_unigram_max_length)
    char_trigram_vector = vectorize_character_trigrams(text,
                                                       num_features=char_trigrams_num_features,
                                                       max_length=char_trigrams_max_length,
                                                       offset=word_unigram_num_features - 1)

    return word_unigram_vector + char_trigram_vector


clean_pattern = re.compile(r"[^a-z0-9 ]")


def clean_phrase(text: str) -> str:
    decoded = unicodedata.normalize("NFKD", text
                                    .replace('ł', 'l')
                                    .replace('Ł', 'L')
                                    )

    return clean_pattern.sub("", decoded.lower()).strip()
