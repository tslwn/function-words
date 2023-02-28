from typing import Optional

from .abstract import AbstractProtocol, Protocol


class NegationNTCProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, separator: str = " ", seed: Optional[int] = None):
        """
            The number of shapes is fixed to be two.
        """
        super().__init__(num_colors, 2, separator, seed)

    def _get_protocol(self) -> Protocol:
        color_map = {color: color_name for color, color_name
                     in zip(self._colors, self._alphabet[:self._num_colors])}

        shape_map = {shape: shape_name for shape, shape_name
                     in zip(self._shapes, self._alphabet[self._num_colors:])}

        protocol: Protocol = {}

        for color in self._colors:
            shape_0 = self._shapes[0]
            shape_1 = self._shapes[1]

            protocol[color, shape_0] = [
                self._separator.join(
                    [color_map[color], shape_map[shape_0]]),
                self._separator.join(
                    [color_map[color], "NOT", shape_map[shape_1]]),
            ]
            protocol[color, shape_1] = [
                self._separator.join(
                    [color_map[color], shape_map[shape_1]]),
                self._separator.join(
                    [color_map[color], "NOT", shape_map[shape_0]]),
            ]

        return protocol
