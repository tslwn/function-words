from typing import Literal, Optional

from .abstract import AbstractProtocol
from .context_sensitive_ntc import ContextSensitiveNTCProtocol
from .diagonal_ntc import DiagonalNTCProtocol
from .english import EnglishProtocol
from .holistic import HolisticProtocol
from .ntc import NTCProtocol
from .order_sensitive_ntc import OrderSensitiveNTCProtocol
from .random import RandomProtocol
from .rotated_ntc import RotatedNTCProtocol
from .tc import TCProtocol

ProtocolName = Literal[
    "context",
    "diagonal",
    "english",
    "holistic",
    "ntc",
    "order",
    "random",
    "rotated",
    "tc",
]


def get_protocol(protocol_name: ProtocolName, num_colors: int, num_shapes: int, separator: str = " ", seed: Optional[int] = None) -> AbstractProtocol:
    if protocol_name == "context":
        return ContextSensitiveNTCProtocol(num_colors, num_shapes, separator, seed)
    if protocol_name == "diagonal":
        return DiagonalNTCProtocol(num_colors, num_shapes, separator, seed)
    if protocol_name == "english":
        return EnglishProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "holistic":
        return HolisticProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "ntc":
        return NTCProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "order":
        return OrderSensitiveNTCProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "random":
        return RandomProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "rotated":
        return RotatedNTCProtocol(num_colors, num_shapes, separator, seed)
    elif protocol_name == "tc":
        return TCProtocol(num_colors, num_shapes, separator, seed)
    else:
        raise NotImplementedError
