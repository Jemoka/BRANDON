import random
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from nltk.corpus import brown
from nltk.corpus import sinica_treebank

from tqdm import tqdm

import wandb

MIDSIZE = 128
LAMBDA = 0.2
BATCH_SIZE = 4
EPOCHS = 3
LR = 3e-3

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

tensify = lambda x: torch.tensor(x)

# Get a bunch of English
english_words = list(set(brown.words()))
english_sents = brown.sents()

# Get a bunch of Chinese
chinese_words = list(set(sinica_treebank.words()))
chinese_sents = sinica_treebank.sents()

# Word dimention size
num_words = len(english_words)+len(chinese_words)

# Create indexing of the words using a collection
# Create auto counting up defaultdict collection
dictionary = defaultdict(lambda:len(dictionary))
# Iterate through words and initialize
for word in english_words + chinese_words:
    dictionary[word]
# Freeze
dictionary = dict(dictionary)
# Reverse it!
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

# Create bags of words
chinese_sents_indexes = [[dictionary[j] for j in i] for i in chinese_sents][:5000]
english_sents_indexes = [[dictionary[j] for j in i] for i in english_sents][:5000]

english_sents_bags = []
chinese_sents_bags = []

# One hot encode and bag
# English bags
for sent in tqdm(english_sents_indexes):
    # Create temp array
    temp = [0 for _ in range(num_words)]
    # Create temp array of stuff
    for word in sent:
        temp[word] += 1
    # append
    english_sents_bags.append(temp)

# Chinese bags
for sent in tqdm(chinese_sents_indexes):
    # Create temp array
    temp = [0 for _ in range(num_words)]
    # Create temp array of stuff
    for word in sent:
        temp[word] += 1
    # append
    chinese_sents_bags.append(temp)

# Combine bags together
input_data = english_sents_bags+chinese_sents_bags
# Create batches
input_data_batches = []
# Create groups of batches
for i in range(0, len(input_data)-BATCH_SIZE):
    input_data_batches.append(input_data[i:i+BATCH_SIZE])



