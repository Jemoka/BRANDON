import torch
import numpy as np

import wandb

from nltk.corpus import brown
from nltk.corpus import sinica_treebank

# Prove Brandon wrong with me, and if we don't
# heck, we get a paper out of it.
#
# Plach of attach
# 1. create a autoencoding mechanism, f(g(x))
#      where null space of g muuuch smaller than
#      the null space of x or f.
# 2. create a discount factor gamma=0.1
# 3. create a loss function
#      gamma*-1*log(g(x)+(1-gamma)*-1*x*log(f(g(x))
#      this is the gamma-scaled sum between the entropy
#      of the middle layer plus reconstruction error
# 4. ok, now, autoencode two target languages. Say
#      chinese and english. Let's see what happens.

# Get a bunch of English
english_words = set([i.lower() for i in brown.words()])
english_words = list(english_words)

# Get a bunch of Chinese
chinese_words = list(set(sinica_treebank.words()))

# Word dimention



