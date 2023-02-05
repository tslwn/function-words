from train.abstract.trainer import AbstractTrainer
from .WikiSemanticContentFinder import WikiSemanticContentFinder


class WikiTrainer(AbstractTrainer):
    _trainer = "wiki"

    def train(self):
        self._semantic_content_finder = WikiSemanticContentFinder(
            self._count, self._window_size)

    def write(self):
        self._semantic_content_finder.write(self._file)
        del self._semantic_content_finder
