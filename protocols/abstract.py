# pyright: reportUnknownMemberType=false

from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import random
import string
from typing import cast, Optional, Tuple, Union

Derivation = Union[str, Tuple["Derivation", "Derivation"]]

Protocol = dict[Derivation, list[str]]


class AbstractProtocol(ABC):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        self._num_colors = num_colors
        self._num_shapes = num_shapes
        self._separator = separator
        self._seed = seed

        self._colors = [f"c{index}" for index in range(num_colors)]
        self._shapes = [f"c{index}" for index in range(num_shapes)]

        random.seed(self._seed)

        self._alphabet = self._get_alphabet()
        self._protocol = self._get_protocol()

    def _get_alphabet(self) -> list[str]:
        letters: set[str] = set()
        num_letters = self._num_colors + self._num_shapes

        char = partial(
            np.random.choice,
            np.array(list(string.ascii_uppercase + string.digits)))

        while len(letters) < num_letters:
            letters |= {"".join([cast(str, char()) for _ in range(np.random.randint(12, 20))])
                        for _ in range(num_letters - len(letters))}

        alphabet = list(letters)
        random.shuffle(alphabet)

        return alphabet

    @abstractmethod
    def _get_protocol(self) -> Protocol:
        pass

    def samples(self, sample_size: int) -> list[str]:
        samples: list[str] = []

        for derivation in random.choices(list(self._protocol.keys()), k=sample_size):
            sample = random.choice(self._protocol[derivation])

            samples.append(sample)

        return samples

    def documents(self, sample_size: int) -> list[list[tuple[str, bool]]]:
        documents: list[list[tuple[str, bool]]] = []

        for sample in self.samples(sample_size):
            document: list[tuple[str, bool]] = []

            for word in sample.split(self._separator):
                document.append((word, False))

            documents.append(document)

        return documents
