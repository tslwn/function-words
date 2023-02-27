
from copy import deepcopy
from itertools import product
import random
from typing import Optional

from .abstract import AbstractProtocol, Protocol


class OrderSensitiveNTCProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        objects = product(self._colors, self._shapes)

        random.shuffle(self._alphabet)
        color_names = deepcopy(self._alphabet)

        random.shuffle(self._alphabet)
        shape_names = deepcopy(self._alphabet)

        color_map = {color: color_name for color, color_name
                     in zip(self._colors, color_names)}

        shape_map = {shape: shape_name for shape, shape_name
                     in zip(self._shapes, shape_names)}

        protocol: Protocol = {}

        for color, shape in objects:
            protocol[(color, shape)] = [
                self._separator.join((color_map[color], shape_map[shape]))
            ]

        return protocol
