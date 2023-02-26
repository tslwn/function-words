import random
from typing import Optional

from .protocols import Protocol


class ProtocolSampler:
    def __init__(self, protocol: Protocol, seed: Optional[int] = None) -> None:
        self._protocol = protocol
        self._derivations = list(protocol.keys())

        random.seed(seed)

    def samples(self, sample_size: int) -> list[str]:
        return [self._protocol[derivation] for derivation in random.choices(self._derivations, k=sample_size)]

    def documents(self, sample_size: int) -> list[list[tuple[str, bool]]]:
        documents: list[list[tuple[str, bool]]] = []

        for sample in self.samples(sample_size):
            document: list[tuple[str, bool]] = []

            for word in sample.split(" "):
                document.append((word, False))

            documents.append(document)

        return documents
