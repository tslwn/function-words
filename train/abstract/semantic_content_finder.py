
# pyright: reportMissingTypeStubs=false,reportUnknownMemberType=false,reportUnknownVariableType=false
from abc import ABC, abstractmethod
import csv
from math import log
from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder
from typing import Dict, List, Optional, Union
from tag.abstract import AbstractTagger, TaggedDocument, TaggedWord


Row = List[Union[bool, float, str]]


class AbstractSemanticContentFinder(ABC):
    def __init__(self, tagger: AbstractTagger, count: Optional[int], window_size: int):
        self._tagger = tagger
        self._count = count
        self._window_size = window_size

    def write(self, file: str):
        with open(file, "w") as csvfile:
            writer = csv.writer(csvfile)

            for row in self._rows:
                writer.writerow(row)

    @property
    def _rows(self) -> List[Row]:
        kl_divergences = self._kl_divergences
        self_informations = self._self_informations
        is_function_words = self._is_function_words

        rows: List[Row] = []

        for (target, kl_divergence) in kl_divergences.items():
            rows.append(
                [target, kl_divergence, self_informations[target], is_function_words[target]])

        rows = sorted(rows, key=lambda row: row[1], reverse=True)

        return rows

    @property
    def _kl_divergences(self) -> Dict[str, float]:
        freq_dist_bigram = FreqDist()

        for document in self._documents:
            bigram_collocation_finder = BigramCollocationFinder.from_words(
                document, window_size=self._window_size)

            freq_dist_bigram.update(bigram_collocation_finder.ngram_fd)

        self.freq_dist_bigram = freq_dist_bigram

        freq_total = 0
        freq_dist_target = FreqDist()
        freq_dist_context = FreqDist()

        for ((target, context), freq) in self.freq_dist_bigram.items():
            freq_total += freq
            freq_dist_target.update({target: freq})
            freq_dist_context.update({context: freq})

        kl_divergences: Dict[str, float] = {}

        for ((target, context), freq) in self.freq_dist_bigram.items():
            assert isinstance(target, str)
            assert isinstance(context, str)

            if target not in kl_divergences:
                kl_divergences[target] = 0.0

            p_context_target = freq_dist_bigram[(target, context)] * freq_total
            p_context = freq_dist_target[target] * freq_dist_context[context]

            kl_divergences[target] += log(p_context_target /
                                          p_context) * p_context

        return kl_divergences

    @property
    def _self_informations(self) -> Dict[str, float]:
        freq_dist = FreqDist(word for (word, _tag) in self._tagged_words)

        return {word: log(freq) for (word, freq) in freq_dist.items()}

    @property
    def _documents(self) -> List[List[str]]:
        documents: List[List[str]] = []

        for tagged_document in self._tagged_documents:
            documents.append([word for (word, _tag) in tagged_document])

        return documents

    @property
    def _is_function_words(self) -> Dict[str, bool]:
        return {word: self._tagger.is_function_word_tag(tag) for (word, tag) in self._tagged_words}

    @property
    @abstractmethod
    def _tagged_words(self) -> List[TaggedWord]:
        pass

    @property
    @abstractmethod
    def _tagged_documents(self) -> List[TaggedDocument]:
        pass
