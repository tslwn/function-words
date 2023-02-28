from typing import Optional

from .abstract import AbstractProtocol, Protocol
from .tc import TCProtocol


class ContextSensitiveNTCProtocol(AbstractProtocol):
    def __init__(self, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None):
        super().__init__(num_colors, num_shapes, separator, seed)

    def _get_protocol(self) -> Protocol:
        protocol: Protocol = {}

        for (derivation, phrases) in TCProtocol(self._num_colors, self._num_shapes, self._separator, self._seed)._protocol.items():
            for phrase in phrases:
                [color, shape] = phrase.split(self._separator)

                protocol[("color", derivation)] = [color + "_"]
                protocol[("shape", derivation)] = [shape + "_"]
                protocol[("both", derivation)] = [phrase]

        return protocol
