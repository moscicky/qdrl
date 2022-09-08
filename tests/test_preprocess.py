from qdrl.preprocess import WordUnigramVectorizer, \
    CharacterTrigramVectorizer, CombinedVectorizer, DictionaryTextVectorizer


# token 0 should be reserved for padding idx
class TestPreprocess:

    def test_word_vectorizer(self):
        vectorizer = WordUnigramVectorizer(num_features=3, max_length=10)
        vectorized = vectorizer.vectorize("green iphone 11 64 gb")

        assert vectorized == [1, 2, 1, 2, 2, 0, 0, 0, 0, 0]

    def test_char_trigram_vectorizer(self):
        vectorizer = CharacterTrigramVectorizer(num_features=3, max_length=20)
        vectorized = vectorizer.vectorize("green iphone 11 64 gb")

        assert vectorized == [2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 0]

    def test_combined(self):
        vectorizer = CombinedVectorizer(word_unigram_num_features=3, word_unigram_max_length=10,
                                        char_trigrams_num_features=3, char_trigrams_max_length=20)
        vectorized = vectorizer.vectorize("green iphone 11 64 gb")

        assert vectorized == [1, 2, 1, 2, 2, 0, 0, 0, 0, 0, 4, 4, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 0]

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
                              7, 9,
                              1, 4, 9, 5, 6, 8]

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
        assert vectorized == [1, 2, 3, 5]

    # ala, ma, kota
    # ala ma, ma kota (oov)
    # ala, la_, a_m(oov), ma_,  a_k (oov)
