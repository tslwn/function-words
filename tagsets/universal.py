from .abstract import AbstractTagset


class UniversalTagset(AbstractTagset):
    @property
    def _function_word_tags(self) -> list[str]:
        return [
            "ADP",
            "CONJ",
            "DET",
            "PRT",
            "PRON",
        ]

    @property
    def _removed_tags(self) -> list[str]:
        return [
            "NUM",
            ".",
            "!",
            "X",
        ]
