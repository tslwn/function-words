# pyright: reportMissingTypeStubs=false,reportPrivateUsage=false,reportUnknownMemberType=false
import nltk
import ssl

# https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")
nltk.download("wordnet")
