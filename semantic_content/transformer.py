# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from math import isnan, log
from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SemanticContentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int = 11) -> None:
        self._window_size = window_size

    index_: int = 0

    indices_: dict[str, int]
    words_: dict[int, str]

    kl_divergences_: dict[int, float]
    is_function_words_: dict[int, bool]

    def fit(self, X: list[list[tuple[str, bool]]]):
        self.indices_ = {}
        self.words_ = {}

        self.is_function_words_ = {}
        self.kl_divergences_ = {}

        # Create uni- and bigram frequency distributions for all documents.
        freq_dist_unigram = FreqDist()
        freq_dist_bigram = FreqDist()

        for document in X:
            words: list[str] = []

            for (word, is_function_word) in document:
                if word not in self.indices_:
                    self.indices_[word] = self.index_
                    self.words_[self.index_] = word

                    self.index_ += 1

                index = self.indices_[word]

                if index not in self.is_function_words_:
                    self.is_function_words_[index] = is_function_word

                if index not in self.kl_divergences_:
                    self.kl_divergences_[index] = 0.0

                # Update the unigram frequency distribution.
                freq_dist_unigram.update({word: 1})

                words.append(word)

            # Update the bigram frequency distribution.
            bigram_collocation_finder = BigramCollocationFinder.from_words(
                words, window_size=self._window_size
            )

            freq_dist_bigram.update(bigram_collocation_finder.ngram_fd)

        # Find the total count of word tokens.
        freq_total = freq_dist_unigram.N()

        for ((word_target, word_context), freq) in freq_dist_bigram.items():
            # Appease the type checker.
            assert isinstance(word_target, str)
            assert isinstance(word_context, str)

            Q = freq * freq_total

            P = freq_dist_unigram[word_target] * \
                freq_dist_unigram[word_context]

            index = self.indices_[word_target]

            self.kl_divergences_[index] += log(Q / P) * P

        return self

    def transform(self, _X: list[list[tuple[str, bool]]]) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
        check_is_fitted(self, "kl_divergences_")
        check_is_fitted(self, "is_function_words_")

        kl_divergences: list[float] = []
        is_function_words: list[int] = []

        for index in range(0, self.index_):
            word = self.words_.get(index, None)
            kl_divergence = self.kl_divergences_.get(index, float("nan"))
            is_function_word = self.is_function_words_.get(index, None)

            if word is not None and not isnan(kl_divergence) and is_function_word is not None:
                kl_divergences.append(kl_divergence)
                is_function_words.append(is_function_word)

        return np.array(kl_divergences).reshape(-1, 1), np.array(is_function_words)

    def get_feature_names_out(self, input_features: None = None) -> list[str]:
        check_is_fitted(self, "words_")

        words: list[str] = []

        for index in range(0, self.index_):
            word = self.words_.get(index, None)
            kl_divergence = self.kl_divergences_.get(index, float("nan"))
            is_function_word = self.is_function_words_.get(index, None)

            if word is not None and not isnan(kl_divergence) and is_function_word is not None:
                words.append(word)

        return words
