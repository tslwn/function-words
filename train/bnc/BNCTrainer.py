from train.abstract.trainer import AbstractTrainer
from .BNCSemanticContentFinder import BNCSemanticContentFinder


class BNCTrainer(AbstractTrainer):
    _trainer = "bnc"

    def train(self):
        self._semantic_content_finder = BNCSemanticContentFinder(
            self._count, self._window_size)

    def write(self):
        self._semantic_content_finder.write(self._file)
        del self._semantic_content_finder
