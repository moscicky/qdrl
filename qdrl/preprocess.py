import abc
import re
import unicodedata
from typing import List
from hashlib import md5


class TextVectorizer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def vectorize(text: str) -> List[int]:
        pass

    def do_vectorize(self, tokens: List[str], num_features: int, max_length: int, offset: int) -> List[int]:
        buckets = num_features - 1
        vectorized = [(int(md5(t.encode('utf-8')).hexdigest(), 16) % buckets) + offset + 1 for t in tokens][:max_length]
        vectorized += [0] * (max_length - len(vectorized))
        return vectorized


class WordUnigramVectorizer(TextVectorizer):
    def __init__(self, num_features: int, max_length: int, offset: int = 0):
        self.num_features = num_features
        self.max_length = max_length
        self.offset = offset

    def vectorize(self, text: str) -> List[int]:
        tokens = text.split(sep=" ")
        return self.do_vectorize(tokens, self.num_features, self.max_length, self.offset)


class CharacterTrigramVectorizer(TextVectorizer):
    def __init__(self, num_features: int, max_length: int, offset: int = 0):
        self.num_features = num_features
        self.max_length = max_length
        self.offset = offset

    def vectorize(self, text: str) -> List[int]:
        trigrams = re.findall(r'(?=(...))', text)
        return self.do_vectorize(trigrams, self.num_features, self.max_length, self.offset)


class CombinedVectorizer(TextVectorizer):
    def __init__(self,
                 word_unigram_num_features: int,
                 word_unigram_max_length: int,
                 char_trigrams_num_features: int,
                 char_trigrams_max_length: int):
        self.word_vectorizer = WordUnigramVectorizer(num_features=word_unigram_num_features,
                                                     max_length=word_unigram_max_length)
        self.char_vecrorizer = CharacterTrigramVectorizer(
            num_features=char_trigrams_num_features,
            max_length=char_trigrams_max_length,
            offset=word_unigram_num_features - 1
        )

    def vectorize(self, text: str) -> List[int]:
        word_unigram_vector = self.word_vectorizer.vectorize(text)
        char_trigram_vector = self.char_vecrorizer.vectorize(text)

        return word_unigram_vector + char_trigram_vector


clean_pattern = re.compile(r"[^a-z0-9 ]")


def clean_phrase(text: str) -> str:
    decoded = unicodedata.normalize("NFKD", text
                                    .replace('ł', 'l')
                                    .replace('Ł', 'L')
                                    )

    return clean_pattern.sub("", decoded.lower()).strip()
