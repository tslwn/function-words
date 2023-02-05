# pyright: reportMissingTypeStubs=false,reportUnknownMemberType=false,reportUnknownVariableType=false
from glob import glob
from os import getcwd
from typing import List, Optional
from nltk.corpus.reader.bnc import BNCCorpusReader
from tag.abstract import TaggedDocument, TaggedWord
from tag.c5 import C5Tagger
from train.abstract.semantic_content_finder import AbstractSemanticContentFinder


class BNCSemanticContentFinder(AbstractSemanticContentFinder):
    def __init__(self, count: Optional[int], window_size: int):
        super().__init__(C5Tagger(), count, window_size)

        self._c5 = True
        self._stem = True

        self._bnc_corpus_reader = BNCCorpusReader(
            root="data/bnc/Texts", fileids=r'[A-K]/\w*/\w*\.xml')

        self._fileids = list(glob(
            getcwd() + "/data/bnc/Texts/**/*.xml", recursive=True))

        if self._count is not None:
            assert self._count > 0
            self._fileids = self._fileids[0:self._count]

    @property
    def _tagged_words(self) -> List[TaggedWord]:
        return self._tagged_words_fileids(self._fileids)

    @property
    def _tagged_documents(self) -> List[TaggedDocument]:
        return [self._tagged_words_fileids([fileid]) for fileid in self._fileids]

    def _tagged_words_fileids(self, fileids: List[str]) -> List[TaggedWord]:
        tagged_words: List[TaggedWord] = []

        for (word, tag) in self._bnc_corpus_reader.tagged_words(fileids=fileids, c5=self._c5, stem=self._stem):
            assert isinstance(word, str)
            assert isinstance(tag, str)

            if not self._tagger.is_removed_tag(tag):
                tagged_words.append((word, tag))

        return tagged_words
