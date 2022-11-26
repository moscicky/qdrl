from qdrl.preprocess import DictionaryTextVectorizer


# token 0 should be reserved for padding idx
class TestPreprocess:
    def test_dicitionary_vectorizer(self):
        dictionary = {
            "ala": 1,
            "ma": 2,
            "kota": 3,
            "la ": 4,
            " ma": 5,
            "ma ": 6,
            "ala ma": 7

        }

        vectorizer = DictionaryTextVectorizer(
            word_unigrams_limit=3, char_trigrams_limit=6, word_bigrams_limit=2, dictionary=dictionary, num_oov_tokens=2
        )

        vectorized = vectorizer.vectorize("ala ma kota zielonego")
        assert vectorized == [1, 2, 3,
                              7, 8,
                              1, 4, 8, 5, 6, 8]

    def test_dicitionary_vectorizer_only_token_ungirams(self):
        dictionary = {
            "ala": 1,
            "ma": 2,
            "kota": 3
        }

        vectorizer = DictionaryTextVectorizer(
            word_unigrams_limit=4, char_trigrams_limit=0, word_bigrams_limit=0, dictionary=dictionary,
            num_oov_tokens=2
        )

        vectorized = vectorizer.vectorize("ala ma kota zielonego 23")
        assert vectorized == [1, 2, 3, 4]

    # ala, ma, kota
    # ala ma, ma kota (oov)
    # ala, la_, a_m(oov), ma_,  a_k (oov)
