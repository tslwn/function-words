# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from argparse import ArgumentParser
import pandas as pd
from train.abstract.corpus import AbstractCorpus
from train.bnc.bnc_corpus import BNCCorpus
from train.wiki.wiki_corpus import WikiCorpus
from train.semantic_content_transformer import SemanticContentTransformer

parser = ArgumentParser()
parser.add_argument("-c", "--corpus", choices=["bnc", "wiki"])
parser.add_argument("-n", "--sample-size", type=float, default=1.0)
parser.add_argument("-w", "--window-size", type=int, default=11)
args = parser.parse_args()

corpus: AbstractCorpus

if args.corpus == "bnc":
    corpus = BNCCorpus(sample_size=args.sample_size)
elif args.corpus == "wiki":
    corpus = WikiCorpus(sample_size=args.sample_size)
else:
    raise NotImplementedError()

semantic_content_transformer = SemanticContentTransformer(
    window_size=args.window_size)

semantic_contents = semantic_content_transformer.fit_transform(
    corpus.documents())

data = pd.DataFrame(data=semantic_contents, columns=[
                    "KL divergence", "Function word"])
data["Word"] = semantic_content_transformer.get_feature_names_out()

path = f"results/{args.corpus}/sample_size_{args.sample_size}_window_size_{args.window_size}.csv"
data.to_csv(path)
