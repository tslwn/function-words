# pyright: reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
import nltk
import ssl
from typing import Literal

from .abstract import AbstractCorpus
from .bnc import BNCCorpus
from .wiki import WikiCorpus

# https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("universal_tagset", quiet=True)
nltk.download("wordnet", quiet=True)


CorpusName = Literal["BNC", "Simple English Wikipedia"]


def get_corpus(corpus_name: CorpusName, seed: int, sample_size: float) -> AbstractCorpus:
    if corpus_name == "BNC":
        return BNCCorpus(seed=seed, sample_size=sample_size)
    elif corpus_name == "Simple English Wikipedia":
        return WikiCorpus(seed=seed, sample_size=sample_size)
    else:
        raise NotImplementedError
