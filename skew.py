# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false
import numpy as np
from scipy.stats import skew
from train.train import get_parameters_str, Parameters, parameter_grid, train
from typing import cast


def get_skew(parameters: Parameters) -> tuple[float, float]:
    x, y = train(parameters)

    kl_divergences_function = np.array([kl_divergence for (
        kl_divergence, is_function_word) in zip(x, y) if is_function_word == True])

    kl_divergences_content = np.array([kl_divergence for (
        kl_divergence, is_function_word) in zip(x, y) if is_function_word == False])

    skew_function = skew(kl_divergences_function)
    assert isinstance(skew_function, float)

    skew_content = skew(kl_divergences_content)
    assert isinstance(skew_content, float)

    return skew_function, skew_content


for parameters in parameter_grid:
    parameters = cast(Parameters, parameters)

    skew_function, skew_content = get_skew(parameters)

    print(get_parameters_str(parameters))
    print(f"Skew (function words) = {skew_function}")
    print(f"Skew (content words) = {skew_content}")
