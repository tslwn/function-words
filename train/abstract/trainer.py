# pyright: reportMissingTypeStubs=false,reportUnknownMemberType=false,reportUnknownVariableType=false
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional


class AbstractTrainer(ABC):
    _trainer: str
    _names = ["Word", "KL divergence", "Self-information", "Is function word"]

    def __init__(self, count: Optional[int], window_size: int):
        self._count = count
        self._window_size = window_size

        self._file = f"results/{self._trainer}/count_{count}_window_size_{window_size}.csv"

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def write(self):
        pass

    def plot(self):
        try:
            data = pd.read_csv(self._file, names=self._names)

            sns.scatterplot(data=data, x="Self-information", y="KL divergence",
                            hue="Is function word")

            plt.show()
        except Exception as exception:
            print(exception)
