from argparse import ArgumentParser
from train.abstract.trainer import AbstractTrainer
from train.bnc.BNCTrainer import BNCTrainer
from train.wiki.WikiTrainer import WikiTrainer

parser = ArgumentParser()
parser.add_argument("-c", "--corpus", choices=["bnc", "wiki"])
parser.add_argument("-n", "--count", type=int)
parser.add_argument("-w", "--window-size", type=int)

parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("--no-train", dest="train", action="store_false")
parser.set_defaults(train=True)

parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("--no-plot", dest="plot", action="store_false")
parser.set_defaults(plot=False)

args = parser.parse_args()

corpus = args.corpus
count = args.count
window_size = args.window_size

train = args.train
plot = args.plot

trainer: AbstractTrainer
if corpus == "bnc":
    trainer = BNCTrainer(count, window_size)
elif corpus == "wiki":
    trainer = WikiTrainer(count, window_size)
else:
    raise NotImplementedError()

if args.train:
    trainer.train()
    trainer.write()

if args.plot:
    trainer.plot()
