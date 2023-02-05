# pyright: reportMissingTypeStubs=false,reportUnknownVariableType=false
from datasets.load import load_dataset
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from typing import cast, List, Optional, TypedDict
from tag.abstract import TaggedDocument, TaggedWord
from tag.penn_treebank import PennTreebankTagger
from train.abstract.semantic_content_finder import AbstractSemanticContentFinder


class Article(TypedDict):
    id: str
    url: str
    title: str
    text: str


class WikiSemanticContentFinder(AbstractSemanticContentFinder):
    def __init__(self, count: Optional[int], window_size: int):
        super().__init__(PennTreebankTagger(), count, window_size)

        self._tagset = None  # Penn Treebank
        self._lemmatizer = WordNetLemmatizer()

        dataset = load_dataset(
            "wikipedia", "20220301.simple", split="train")

        articles: List[Article] = []
        index = 0

        for article in dataset:
            articles.append(cast(Article, article))
            index += 1

            if count is not None and index >= count:
                break

        self._articles = articles

    @property
    def _tagged_words(self) -> List[TaggedWord]:
        return [(word, tag) for document in self._tagged_documents for (word, tag) in document]

    @property
    def _tagged_documents(self) -> List[TaggedDocument]:
        return [self._tagged_words_article(article) for article in self._articles]

    def _tagged_words_article(self, article: Article) -> List[TaggedWord]:
        tagged_words: List[TaggedWord] = []

        for (word, tag) in pos_tag(word_tokenize(article["text"]), tagset=self._tagset):
            assert isinstance(word, str)

            if not self._tagger.is_removed_tag(tag):
                lemma = self._lemmatizer.lemmatize(word.lower())
                tagged_words.append((lemma, tag))

        return tagged_words
