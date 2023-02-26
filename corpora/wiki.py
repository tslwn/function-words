
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false

from datasets.load import load_dataset
from datasets.utils.logging import set_verbosity_error
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
from typing import cast, Optional, TypedDict

from corpora.abstract import AbstractCorpus
from tagsets.penn_treebank import PennTreebankTagset

# Suppress `datasets` log messages.
set_verbosity_error()


class Article(TypedDict):
    id: str
    url: str
    title: str
    text: str


class WikiCorpus(AbstractCorpus):
    def __init__(self, seed: Optional[int] = None, sample_size: float = 1.0) -> None:
        super().__init__(seed, sample_size)

        self._tagset = PennTreebankTagset()

        # Load all of the articles
        dataset = load_dataset(
            "wikipedia", "20220301.simple", split="train")

        articles: list[Article] = []
        article_count = 0

        for article in dataset:
            articles.append(cast(Article, article))
            article_count += 1

        # Find the number of articles to randomly sample.
        article_count = int(article_count * self._sample_size)

        # Initialise the random number generator.
        random.seed(seed)

        # Randomly sample the articles.
        self._articles = random.sample(articles, article_count)

    _article_index = 0

    def documents(self) -> list[list[tuple[str, bool]]]:
        lemmatizer = WordNetLemmatizer()

        documents: list[list[tuple[str, bool]]] = []

        for article in self._articles:
            document: list[tuple[str, bool]] = []

            for (word, tag) in pos_tag(word_tokenize(article["text"]), tagset=None):
                # Appease the type checker.
                assert isinstance(word, str)

                if not self._tagset.is_removed_tag(tag):
                    lemma = lemmatizer.lemmatize(word.lower())
                    document.append(
                        (lemma, self._tagset.is_function_word_tag(tag)))

            documents.append(document)

        return documents
