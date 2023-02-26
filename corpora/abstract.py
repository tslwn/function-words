from abc import ABC, abstractmethod
from typing import Optional


class AbstractCorpus(ABC):
    def __init__(self, seed: Optional[int] = None, sample_size: float = 1.0) -> None:
        self._seed = seed
        self._sample_size = sample_size

    @abstractmethod
    def documents(self) -> list[list[tuple[str, bool]]]:
        pass
