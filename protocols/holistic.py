from itertools import product
import random
from typing import Optional

from .abstract import AbstractProtocol, Protocol


class HolisticProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        objects = product(self._colors, self._shapes)

        object_names = list(product(self._alphabet, self._alphabet))
        random.shuffle(object_names)

        protocol: Protocol = {}

        for (color, shape), name in zip(objects, object_names):
            protocol[(color, shape)] = [self._separator.join(name)]

        return protocol
