from itertools import product
from typing import Optional

from .abstract import AbstractProtocol, Protocol


class TCProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        color_map = {color: color_name for color, color_name
                     in zip(self._colors, self._alphabet[:self._num_colors])}

        shape_map = {shape: shape_name for shape, shape_name
                     in zip(self._shapes, self._alphabet[self._num_colors:])}

        protocol: Protocol = {}

        for color, shape in product(self._colors, self._shapes):
            protocol[color, shape] = [
                self._separator.join(
                    [color_map[color], shape_map[shape]]),
                self._separator.join(
                    [shape_map[shape], color_map[color]])
            ]

        return protocol
