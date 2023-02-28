# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from datetime import datetime
import pandas
from scipy import stats
from sklearn.model_selection import ParameterGrid
from typing import cast, NamedTuple

from corpora import CorpusName
from experiment_1_results import get_parameter_values, get_result, Parameters
from semantic_content.pipeline import ScalerName

parameter_grid = ParameterGrid({
    "corpus_name": [
        "BNC",
        "Simple English Wikipedia"
    ],
    "seed": list(range(1)),
    "sample_size": [0.001, 0.01, 0.1],
    "window_size": [11],
    "scaler_name": [
        "standard",
    ]
})


Statistics = NamedTuple("Statistics", [
    ("corpus_name", CorpusName),
    ("sample_size", float),
    ("window_size", int),
    ("scaler_name", ScalerName),
    ("seed", int),
    ("word_class", str),
    ("n", int),
    ("mean", float),
    ("variance", float),
    ("skewness", float),
    ("kurtosis", float),
])

TestResults = NamedTuple("TestResults", [
    ("corpus_name", CorpusName),
    ("sample_size", float),
    ("window_size", int),
    ("scaler_name", ScalerName),
    ("seed", int),
    ("p_kolmogorov_smirnov", float),
    ("p_cramervonmises", float),
    ("p_epps_singleton", float),
    ("p_anderson", float),
])


if __name__ == "__main__":
    statistics: list[Statistics] = []
    test_results: list[TestResults] = []

    for parameters in parameter_grid:
        parameters = cast(Parameters, parameters)
        corpus_name, seed, sample_size, window_size, scaler_name = get_parameter_values(
            parameters)

        x, y = get_result(parameters)

        n, _minmax, mean, variance, skewness, kurtosis = stats.describe(x)
        statistics.append(Statistics(corpus_name, sample_size, window_size,
                          scaler_name, seed, "all", n, mean, variance, skewness, kurtosis))

        x_f = [value for (value, f) in zip(x, y) if f]
        n, _minmax, mean, variance, skewness, kurtosis = stats.describe(x_f)
        statistics.append(Statistics(corpus_name, sample_size, window_size,
                          scaler_name, seed, "function", n, mean, variance, skewness, kurtosis))

        x_c = [value for (value, f) in zip(x, y) if not f]
        n, _minmax, mean, variance, skewness, kurtosis = stats.describe(x_c)
        statistics.append(Statistics(corpus_name, sample_size, window_size,
                          scaler_name, seed, "content", n, mean, variance, skewness, kurtosis))

        _, p_kolmogorov_smirnov = stats.ks_2samp(x_f, x_c)

        result_cramervonmises = stats.cramervonmises_2samp(x_f, x_c)

        result_epps_singleton = stats.epps_singleton_2samp(x_f, x_c)

        _, _, p_anderson = stats.anderson_ksamp([x_f, x_c])

        test_results.append(TestResults(corpus_name, sample_size, window_size, scaler_name, seed, p_kolmogorov_smirnov,
                            float(result_cramervonmises.pvalue), result_epps_singleton.pvalue, p_anderson))

    df_statistics = pandas.DataFrame.from_records(
        statistics, columns=Statistics._fields)
    df_test_results = pandas.DataFrame.from_records(
        test_results, columns=TestResults._fields)

    dt = datetime.now().strftime("%Y%m%d%H%M%S")

    df_statistics.to_csv(f"results/{dt}_experiment_1_statistics.csv")
    df_test_results.to_csv(f"results/{dt}_experiment_1_test_results.csv")
