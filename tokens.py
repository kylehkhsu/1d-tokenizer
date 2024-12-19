import numpy as np
import ipdb
from nltk import bigrams, FreqDist
tokens_path = "/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/titok_l32/train_tokens.npy"

tokens = np.load(tokens_path)

all_bigrams = [bigram for sequence in tokens for bigram in bigrams(sequence)]

bigram_freq = FreqDist(all_bigrams)

ipdb.set_trace()