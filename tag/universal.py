from typing import List
from .abstract import AbstractTagger


class UniversalTagger(AbstractTagger):
    @property
    def _function_word_tags(self) -> List[str]:
        return [
            "ADP",
            "CONJ",
            "DET",
            "PRT",
            "PRON",
        ]

    @property
    def _removed_tags(self) -> List[str]:
        return [
            "NUM",
            ".",
            "!",
            "X",
        ]
