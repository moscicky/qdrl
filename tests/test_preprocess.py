from qdrl.preprocess import vectorize_word, vectorize_character_trigrams, vectorize_combined


# token 0 should be reserved for padding idx
class TestPreprocess:

    def test_word_vectorizer(self):
        vectorized = vectorize_word("green iphone 11 64 gb", num_features=3, max_length=10)

        assert vectorized == [1, 2, 1, 2, 2, 0, 0, 0, 0, 0]

    def test_char_trigram_vectorizer(self):
        vectorized = vectorize_character_trigrams("green iphone 11 64 gb", num_features=3, max_length=20)

        assert vectorized == [2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 0]

    def test_combined(self):
        vectorized = vectorize_combined("green iphone 11 64 gb",
                           word_unigram_num_features=3, word_unigram_max_length=10,
                           char_trigrams_num_features=3, char_trigrams_max_length=20)

        assert vectorized == [1, 2, 1, 2, 2, 0, 0, 0, 0, 0, 4, 4, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 0]

