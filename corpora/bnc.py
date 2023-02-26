
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import glob
from nltk.corpus.reader.bnc import BNCCorpusReader
import os
import random
from typing import Optional

from .abstract import AbstractCorpus
from tagsets.c5 import C5Tagset


class BNCCorpus(AbstractCorpus):
    def __init__(self, seed: Optional[int] = None, sample_size: float = 1.0) -> None:
        super().__init__(seed, sample_size)

        self._tagset = C5Tagset()

        # Find the file paths of all of the documents.
        fileids = glob.glob(
            os.getcwd() + "/data/bnc/Texts/**/*.xml", recursive=True)

        # Find the number of file paths to randomly sample.
        fileid_count = int(len(fileids) * self._sample_size)

        # Initialise the random number generator.
        random.seed(seed)

        # Randomly sample the file paths.
        self._fileids = random.sample(fileids, fileid_count)

        # Initialise the corpus reader.
        self._reader = BNCCorpusReader(
            root="data/bnc/Texts", fileids=self._fileids)

    _fileid_index = 0

    def documents(self) -> list[list[tuple[str, bool]]]:
        documents: list[list[tuple[str, bool]]] = []

        for fileid in self._fileids:
            document: list[tuple[str, bool]] = []

            for (word, tag) in self._reader.tagged_words(fileids=[fileid], c5=True, stem=True):
                # Appease the type checker.
                assert isinstance(word, str)
                assert isinstance(tag, str)

                if not self._tagset.is_removed_tag(tag):
                    document.append(
                        (word, self._tagset.is_function_word_tag(tag)))

            documents.append(document)

        return documents
