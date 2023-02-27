from typing import Optional

from .abstract import AbstractProtocol, Protocol


class NTCProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        protocol: Protocol = {}

        for index_color, color in enumerate(self._colors):
            for index_shape, shape in enumerate(self._shapes):
                word_1 = self._alphabet[(index_color - index_shape) %
                                        (self._num_colors + self._num_shapes)]

                word_2 = self._alphabet[(index_color + index_shape) %
                                        (self._num_colors + self._num_shapes)]

                protocol[color, shape] = [word_1 + self._separator + word_2]

        return protocol
