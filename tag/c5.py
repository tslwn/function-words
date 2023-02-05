from typing import List
from .abstract import AbstractTagger


class C5Tagger(AbstractTagger):
    @property
    def _function_word_tags(self) -> List[str]:
        return [
            "AT0",
            "AVP",
            "CJC",
            "CJS",
            "CJT",
            "DPS",
            "DT0",
            "DTQ",
            "EX0",
            "ITJ",
            "PNI",
            "PNP",
            "PNQ",
            "PNX",
            "POS",
            "PRF",
            "PRP",
            "TO0",
            "VBB",
            "VBD",
            "VBG",
            "VBI",
            "VBN",
            "VBZ",
            "VDB",
            "VDD",
            "VDG",
            "VDI",
            "VDN",
            "VDZ",
            "VHB",
            "VHD",
            "VHG",
            "VHI",
            "VHN",
            "VHZ",
            "VM0",
            "XX0"
        ]

    @property
    def _removed_tags(self) -> List[str]:
        return [
            "CRD",
            "ORD",
            "PUL",
            "PUN",
            "PUQ",
            "PUR",
            "UNC",
            "ZZ0"
        ]
