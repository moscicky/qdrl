import abc
import json
import re
from typing import List, Dict
import unicodedata
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


class DictionaryTextVectorizer(TextVectorizer):
    def __init__(self,
                 word_unigrams_limit: int,
                 char_trigrams_limit: int,
                 word_bigrams_limit: int,
                 dictionary: Dict[str, int],
                 num_oov_tokens: int):
        self.dictionary = dictionary
        self.word_unigrams_limit = word_unigrams_limit
        self.char_trigrams_limit = char_trigrams_limit
        self.word_bigrams_limit = word_bigrams_limit
        self.num_tokens = len(dictionary)
        self.num_oov_tokens = num_oov_tokens

    def vectorize(self, text: str) -> List[int]:
        return self.vectorize_word_unigram(text) + self.vectorize_word_bigram(text) + self.vectorize_char_trigram(text)

    def pad(self, arr: List[int], max_size: int) -> List[int]:
        arr = arr[:max_size]
        arr += [0] * (max_size - len(arr))
        return arr

    def vectorize_oov(self, token: str) -> int:
        return (int(md5(token.encode('utf-8')).hexdigest(), 16) % (self.num_oov_tokens - 1) + self.num_tokens + 1)

    def vectorize_from_dict(self, tokens: List[str]) -> List[int]:
        return [self.dictionary[token] if token in self.dictionary else self.vectorize_oov(token) for token in
                tokens]

    def vectorize_word_unigram(self, text: str) -> List[int]:
        if self.word_unigrams_limit == 0:
            return []
        word_unigrams = text.split(sep=" ")
        vectorized = self.vectorize_from_dict(word_unigrams)
        return self.pad(vectorized, max_size=self.word_unigrams_limit)

    def vectorize_word_bigram(self, text: str) -> List[int]:
        if self.word_bigrams_limit == 0:
            return []
        word_unigrams = text.split(sep=" ")
        word_bigrams = []
        if len(word_unigrams) > 1:
            word_bigrams = [f"{w1} {w2}" for w1, w2 in zip(word_unigrams[:len(word_unigrams) - 1], word_unigrams[1:])]
        vectorized = self.vectorize_from_dict(word_bigrams)
        return self.pad(vectorized, max_size=self.word_bigrams_limit)

    def vectorize_char_trigram(self, text: str) -> List[int]:
        if self.char_trigrams_limit == 0:
            return []
        char_trigrams = re.findall(r'(?=(...))', text)
        vectorized = self.vectorize_from_dict(char_trigrams)
        return self.pad(vectorized, max_size=self.char_trigrams_limit)


class DictionaryLoaderTextVectorizer(DictionaryTextVectorizer):

    def __init__(self,
                 dictionary_path: str,
                 word_unigrams_limit: int,
                 char_trigrams_limit: int,
                 word_bigrams_limit: int,
                 num_oov_tokens: int):
        super(DictionaryLoaderTextVectorizer, self).__init__(
            word_bigrams_limit=word_bigrams_limit,
            char_trigrams_limit=char_trigrams_limit,
            word_unigrams_limit=word_unigrams_limit,
            dictionary=self.load_dictionary(dictionary_path),
            num_oov_tokens=num_oov_tokens
        )

    def load_dictionary(self, path) -> Dict[str, int]:
        dictionary = {}
        with open(f"{path}/dictionary.json") as f:
            for line in f:
                json_line = json.loads(line)
                token = json_line["token"]
                token_id = json_line["token_id"]
                dictionary[token] = token_id
        return dictionary


class WordUnigramVectorizer(TextVectorizer):
    def __init__(self, num_features: int, max_length: int, offset: int = 0):
        self.num_features = num_features
        self.max_length = max_length
        self.offset = offset

    def vectorize(self, text: str) -> List[int]:
        tokens = text.split(sep=" ")
        return self.do_vectorize(tokens, self.num_features, self.max_length, self.offset)


clean_pattern = re.compile(r"[^a-z0-9 ]")


def clean_phrase(text: str) -> str:
    decoded = unicodedata.normalize("NFKD", text
                                    .replace('ł', 'l')
                                    .replace('Ł', 'L')
                                    )

    return clean_pattern.sub("", decoded.lower()).strip()


class CategoricalFeatureMapper:
    def __init__(self,
                 dictionary_path: str,
                 num_oov_features: int
                 ):

        self.dictionary = self.load_dictionary(dictionary_path)
        self.num_known_features = len(self.dictionary)
        self.num_oov_features = num_oov_features

    def load_dictionary(self, path) -> Dict[str, int]:
        dictionary = {}
        with open(f"{path}/dictionary.json") as f:
            for line in f:
                json_line = json.loads(line)
                value = json_line["value"]
                value_id = json_line["value_id"]
                dictionary[value] = value_id
        return dictionary

    def map(self, value: str) -> int:
        if value in self.dictionary:
            return self.dictionary[value]
        else:
            return int(md5(value.encode('utf-8')).hexdigest(), 16) % (
                        self.num_oov_features - 1) + self.num_known_features + 1
