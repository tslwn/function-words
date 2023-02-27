from itertools import product
import random
from typing import Optional

from .abstract import AbstractProtocol, Protocol


class RandomProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        protocol: Protocol = {}

        for color, shape in product(self._colors, self._shapes):
            protocol[(shape, color)] = [
                self._separator.join(
                    [random.choice(self._alphabet), random.choice(self._alphabet)])
            ]

        return protocol
