"""
Adapted from
https://github.com/tomekkorbak/measuring-non-trivial-compositionality/tree/master
"""

# pyright: reportUnknownMemberType=false

from copy import deepcopy
from functools import partial
from itertools import product
import numpy as np
import random
import string
from typing import cast, Tuple, Union

Derivation = Union[str, Tuple["Derivation", "Derivation"]]
Protocol = dict[Derivation, str]

SEPARATOR = " "


def generate_keys(num_keys: int) -> set[str]:
    keys: set[str] = set()

    char = partial(
        np.random.choice,
        np.array(list(string.ascii_uppercase + string.digits)))

    while len(keys) < num_keys:
        keys |= {"".join([cast(str, char()) for _ in range(np.random.randint(12, 20))])
                 for _ in range(num_keys - len(keys))}

    return keys


def generate_elements(num_colors: int, num_shapes: int) -> tuple[list[str], list[str], list[str]]:
    colors = [f"c{i}" for i in range(num_colors)]
    shapes = [f"s{i}" for i in range(num_shapes)]

    alphabet = list(generate_keys(num_colors + num_shapes))
    random.shuffle(alphabet)

    return alphabet, colors, shapes


def get_english_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    color_map = {color: color_name for color, color_name
                 in zip(colors, alphabet[:num_colors])}

    shape_map = {shape: shape_name for shape, shape_name
                 in zip(shapes, alphabet[num_colors:])}

    protocol: Protocol = {}

    for color, shape in product(colors, shapes):
        protocol[color, shape] = SEPARATOR.join([
            color_map[color], "AND", shape_map[shape]])

    return protocol


def get_tc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    color_map = {color: color_name for color, color_name
                 in zip(colors, alphabet[:num_colors])}

    shape_map = {shape: shape_name for shape, shape_name
                 in zip(shapes, alphabet[num_colors:])}

    protocol: Protocol = {}

    for color, shape in product(colors, shapes):
        protocol[(color, shape)] = SEPARATOR.join(
            (color_map[color], shape_map[shape]))

    return protocol


def get_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    protocol: Protocol = {}

    for i, color in enumerate(colors):
        for j, shape in enumerate(shapes):
            word_1 = alphabet[(i - j) % (num_colors + num_shapes)]
            word_2 = alphabet[(i + j) % (num_colors + num_shapes)]
            protocol[color, shape] = word_1 + SEPARATOR + word_2

    return protocol


def get_holistic_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    objects = product(colors, shapes)

    object_names = list(product(alphabet, alphabet))
    random.shuffle(object_names)

    protocol: Protocol = {}

    for (color, shape), name in zip(objects, object_names):
        protocol[(color, shape)] = SEPARATOR.join(name)

    return protocol


def get_random_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    objects = product(colors, shapes)

    protocol: Protocol = {}

    for color, shape in objects:
        protocol[(shape, color)] = SEPARATOR.join(
            [random.choice(alphabet), random.choice(alphabet)])

    return protocol


def get_order_sensitive_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    objects = product(colors, shapes)

    random.shuffle(alphabet)
    color_names = deepcopy(alphabet)

    random.shuffle(alphabet)
    shape_names = deepcopy(alphabet)

    color_map = {color: color_name for color, color_name
                 in zip(colors, color_names)}

    shape_map = {shape: shape_name for shape, shape_name
                 in zip(shapes, shape_names)}

    mapping: Protocol = {}

    for color, shape in objects:
        mapping[(color, shape)] = SEPARATOR.join(
            (color_map[color], shape_map[shape]))

    return mapping


def get_context_sensitive_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    tc_protocol = get_tc_protocol(num_colors, num_shapes)

    protocol: Protocol = {}

    for (derivation, phrase) in tc_protocol.items():
        [color, shape] = phrase.split(SEPARATOR)

        protocol[("color", derivation)] = color + "_"
        protocol[("shape", derivation)] = shape + "_"
        protocol[("both", derivation)] = phrase

    return protocol


def get_diagonal_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    protocol: Protocol = {}

    for i, color in enumerate(colors):
        for j, shape in enumerate(shapes):
            word_1 = alphabet[i + j]
            word_2 = alphabet[j if i + j < num_colors else num_colors - i - 1]
            protocol[color, shape] = word_1 + SEPARATOR + word_2

    return protocol


def get_rotated_ntc_protocol(num_colors: int, num_shapes: int) -> Protocol:
    alphabet, colors, shapes = generate_elements(num_colors, num_shapes)

    protocol: Protocol = {}

    for i, color in enumerate(colors):
        for j, shape in enumerate(shapes):
            word_1 = alphabet[i - j + num_shapes]
            word_2 = alphabet[j + i]
            protocol[color, shape] = word_1 + SEPARATOR + word_2

    return protocol


def get_protocol(protocol_name: str, num_colors: int, num_shapes: int) -> Protocol:
    if protocol_name == "english":
        return get_english_protocol(num_colors=num_colors, num_shapes=num_shapes)
    if protocol_name == "diagonal":
        return get_diagonal_ntc_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "holistic":
        return get_holistic_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "ntc":
        return get_ntc_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "order":
        return get_order_sensitive_ntc_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "random":
        return get_random_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "rotated":
        return get_rotated_ntc_protocol(num_colors=num_colors, num_shapes=num_shapes)
    elif protocol_name == "tc":
        return get_tc_protocol(num_colors=num_colors, num_shapes=num_shapes)
    else:
        raise NotImplementedError
