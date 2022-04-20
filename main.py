from math import gamma
import random
import pickle
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from nltk.tokenize.toktok import ToktokTokenizer

from tqdm import tqdm

import wandb

TRAINING=True

# Create the config
config = {
    "midsize": 1024,
    "gamma": 0,
    "batch_size": 4,
    "epochs": 4,
    "lr": 1e-2,
}

if TRAINING:
    run = wandb.init(project="BRANDON", entity="jemoka", config=config, mode="disabled")
    # run = wandb.init(project="BRANDON", entity="jemoka", config=config)
else:
    run = wandb.init(project="BRANDON", entity="jemoka", config=config, mode="disabled")

config = run.config

# Unroll out the constants
MIDSIZE = config.midsize
GAMMA = config.gamma
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LR = config.lr

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Prove Brandon wrong with me, and if we don't
# heck, we get a paper out of it.
#
# Plach of attach
# 1. create a autoencoding mechanism, f(g(x))
#      where null space of g muuuch smaller than
#      the null space of x or f.
# 2. train an autoencoder in the same manner as
#      described in (P et al. 2014). Sidenote:
#      sorry, that's literally their name.
#      Sarath Chandar A.P.
# 3. abalate. Figure what the median activations
#      are doing---i.e. activating a single middle
#      layer would result in what changes? Also
#      what words result in more of those changes?

tensify = lambda x: torch.tensor(x).float()

# Load in the data!
raw_data_es = []
raw_data_en = []

with open("./data/europarl-es_en/europarl-v7.es-en.en", "r") as df:
    raw_data_en = [i.strip() for i in df.readlines()]

with open("./data/europarl-es_en/europarl-v7.es-en.es", "r") as df:
    raw_data_es = [i.strip() for i in df.readlines()]

# Tokenize them data!
# Instantiate a toktok
tokenizer = ToktokTokenizer()
# Tokenize each one with tqdm tracking progress
tokenized_data_es = []
for i in tqdm(raw_data_es[:10000]):
    tokenized_data_es.append(tokenizer.tokenize(i))
tokenized_data_en = []
for i in tqdm(raw_data_en[:10000]):
    tokenized_data_en.append(tokenizer.tokenize(i))

# Build the twoway dicts
spanish_dict = defaultdict(lambda: len(spanish_dict))
english_dict = defaultdict(lambda: len(english_dict))

for i in tokenized_data_es:
    for j in i:
        spanish_dict[j]

for i in tokenized_data_en:
    for j in i:
        english_dict[j]

# Freeze wordlist
english_dict = dict(english_dict)
spanish_dict = dict(spanish_dict)

# Reverse wordlist
english_dict_inv = dict(zip(english_dict.values(), english_dict.keys()))
spanish_dict_inv = dict(zip(spanish_dict.values(), spanish_dict.keys()))

# Get num words
num_words_es = len(spanish_dict)
num_words_en = len(english_dict)

# Creating binary bags of words
# English...
input_data_en = []
for i in tqdm(tokenized_data_en):
    # Create a temporary zeros array with words
    temp = torch.zeros(num_words_en)
    # For every word, set it as being activated
    # if used
    for word in i:
        temp[english_dict[word]] = 1
    # Append!
    input_data_en.append(temp)

# Spanish...
input_data_es = []
for i in tqdm(tokenized_data_es):
    # Create a temporary zeros array with words
    temp = torch.zeros(num_words_es)
    # For every word, set it as being activated
    # if used
    for word in i:
        temp[spanish_dict[word]] = 1
    # Append!
    input_data_es.append(temp)

# Create input data batches
input_data_batches = []
for i in range(0,len(input_data_en)-BATCH_SIZE,BATCH_SIZE):
    # For each batch, append the pairwise en es
    input_data_batches.append((torch.stack(input_data_es[i:i+BATCH_SIZE]),
                               torch.stack(input_data_en[i:i+BATCH_SIZE])))

# Temp dump data
# with open("./data/europarl-es_en/tokenized.bin", "wb") as df:
#     data = {}
#     data["english_dict"] = english_dict 
#     data["spanish_dict"] = spanish_dict 
#     data["tokenized_data_es"] = tokenized_data_es 
#     data["tokenized_data_en"] = tokenized_data_en 
#     data["num_words_es"] = num_words_es
#     data["num_words_en"] = num_words_en
#     data["input_data_en"] = input_data_en
#     data["input_data_es"] = input_data_es
#     data = pickle.dump(data, df)

# Create the model!
class Autoencoder(nn.Module):

    def __init__(self, vocab_size_in:int, vocab_size_out:int, midsize:int) -> None:
        super().__init__()

        self.vocab_size_in = vocab_size_in
        self.vocab_size_out = vocab_size_out
        self.midsize = midsize

        self.in_layer = nn.Linear(vocab_size_in, midsize)
        self.out_layer = nn.Linear(midsize, vocab_size_out)

    def forward(self, x,label=None) -> dict:
        # Generate results
        encoded_result = F.sigmoid(self.in_layer(x))
        output_result = F.relu(self.out_layer(encoded_result))

        # Return final loss
        return {"logits": encoded_result,
                        # reconstruction loss is loss
                "loss": torch.mean((output_result-label)**2)}

    def decode(self, x) -> any:
        # Decode and return
        return F.relu(self.out_layer(x))

if TRAINING:
    # Instatiate a model
    model = Autoencoder(num_words_es, num_words_en, MIDSIZE).to(DEVICE)
    model.train()
    run.watch(model)

    # Instantiate an optimizer!
    optim = Adam(model.parameters(), lr=LR)

    # For every batch, train!
    for i in range(EPOCHS):
        random.shuffle(input_data_batches)
        print(f"Training epoch {i}")
        # Iterate through batches
        for e, (i,o) in enumerate(tqdm(input_data_batches)):
            # Pass data through
            res = model(i.to(DEVICE), label=o.to(DEVICE))
            # Backpropergate! Ha!
            res["loss"].backward()
            # Step!
            optim.step()
            optim.zero_grad()

            # log?
            if e % 10 == 0:
                wandb.log({"loss": res["loss"].cpu().item()})

    # Save the model
    torch.save(model, f"./models/{run.name}")

else:
    # load the model
    model = torch.load("./models/trim-totem-4")
    
    # instantiate model
    model.eval()

    # load stuff
    def getbin(word):
        # get the id
        word_id = dictionary[word]
        # create temp array
        temp = [0 for _ in range(num_words)]
        # set positive result as 1
        temp[word_id] = 1

        return tensify(temp)

    # breakpoint?
    breakpoint()

# quantumish — Today at 3:17 PM
# "coding" - jack, again
# ghatch — Today at 3:19 PM
# Action Research!
# not
# like
# programming
