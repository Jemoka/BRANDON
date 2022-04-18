from math import gamma
import random
import pickle
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from nltk.corpus import brown
from nltk.corpus import sinica_treebank

from tqdm import tqdm

import wandb

TRAINING=True

# Create the config
config = {
    "midsize": 128,
    "gamma": 0,
    "batch_size": 4,
    "epochs": 4,
    "lr": 1e-3,
}

if TRAINING:
    # run = wandb.init(project="BRANDON", entity="jemoka", config=config, mode="disabled")
    run = wandb.init(project="BRANDON", entity="jemoka", config=config)
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
# 2. create a shift factor gamma=0.1
# 3. create a loss function
#      (1-gamma)*rec_loss+(gamma)*-1*x*log(f(g(x))
#      this is the gamma-scaled sum between the entropy
#      of the middle layer plus reconstruction error
# 4. ok, now, autoencode two target languages. Say
#      chinese and english. Let's see what happens.

tensify = lambda x: torch.tensor(x).float()

# Get a bunch of English
english_words = list(set(brown.words()))
english_sents = brown.sents()

# Get a bunch of Chinese
chinese_words = list(set(sinica_treebank.words()))
chinese_sents = sinica_treebank.sents()

# Word dimention size
num_words = len(english_words)+len(chinese_words)

# Load in data
with open("./data/dataset.dat", "rb") as df:
    english_sents_bags, chinese_sents_bags = pickle.load(df)

# Combine bags together
input_data = english_sents_bags+chinese_sents_bags
# Shuffle the data
random.shuffle(input_data)
# Create batches
input_data_batches = []
# Create groups of batches
for i in range(0, len(input_data)-BATCH_SIZE):
    input_data_batches.append(input_data[i:i+BATCH_SIZE])

# Create the model
class Autoencoder(nn.Module):

    def __init__(self, vocab_size:int, midsize:int, gamma:float=0.2, epsilon:float=1e-7) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.midsize = midsize
        self.gamma = gamma
        self.epsilon = epsilon

        self.in_layer = nn.Linear(vocab_size, midsize)
        self.out_layer = nn.Linear(midsize, vocab_size)

    def forward(self, x) -> dict:
        # Generate results
        encoded_result = F.relu(self.in_layer(x))
        output_result = F.relu(self.out_layer(encoded_result))

        # Create reconstruction loss
        rec_loss = torch.mean((output_result-x)**2)
        entropy = torch.mean(-1*torch.log(encoded_result+self.epsilon)*(encoded_result+self.epsilon))

        # Give gamma
        gamma = self.gamma

        # Return final loss
        return {"logits": encoded_result,
                "loss": (1-self.gamma)*rec_loss + gamma*entropy}

    def decode(self, x) -> any:
        # Decode and return
        return F.relu(self.out_layer(x))

if TRAINING:
    # Instatiate a model
    model = Autoencoder(num_words, MIDSIZE, GAMMA).to(DEVICE)
    model.train()
    run.watch(model)

    # Instantiate an optimizer!
    optim = Adam(model.parameters(), lr=LR)

    # For every batch, train!
    for i in range(EPOCHS):
        random.shuffle(input_data_batches)
        print(f"Training epoch {i}")
        # Iterate through batches
        for e, batch in enumerate(tqdm(input_data_batches)):
            # Pass data through
            res = model(tensify(batch).to(DEVICE))
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
