---
title: 'How to scrape reddit to create your own dataset'
date: 2020-02-25
permalink: /posts/2020/02/scrape-reddit/
tags:
  - nlp
  - code
---

A lot of subreddits have specific submission/comment formats which can make
for interesting NLP datasets. In this post, I'll show you how to create your own
dataset from [reddit.com/r/AmITheAsshole](https://www.reddit.com/r/AmITheAsshole).

{% include toc %}

## Background
As part of the [ImplementAI 2019 hackathon](https://implementai-2019.devpost.com/) 
I wanted to build [something fun](https://devpost.com/software/implementaita/) 
with the [AllenNLP](https://allennlp.org/) library. 
I decided to scrape my own dataset from 
[reddit.com/r/AmItheAsshole/](https://www.reddit.com/r/AmItheAsshole/).

On this subreddit, people post submissions which describe a situation they are in, 
and they ask whether or not they are the asshole in that situation. Then, people
reply with their opinion which other people can vote on. Normally, a submission is 
prefixed with `[AITA]`, and commenters prefix their responses with 
`NTA` for "not the asshole", 
`YTA` for "you're the asshole", 
`ESH` for "everyone sucks here",
`NAH` for "no assholes here", and
`INFO` for "more info needed".

My goal was to create a dataset where, given a submission title and description,
the target output is one of the 5 labels described above. 
If I scrape submissions and their comments, I can extract the most "likely" label 
based on the cumulative scores of comments for each label. 

## Code
### Which library to use
I had previously used [praw](https://github.com/praw-dev/praw) to scrape reddit 
live, however this option requires credentials to use reddit’s API and the 
rate-limits can be restrictive if you’re trying to scrape a large historic dataset.

Instead, I opted to use [psaw](https://github.com/dmarx/psaw) which wraps 
the pushshift.io API and is much more forgiving for scraping larger historic datasets.  

### Pre-tokenization
If we were to naively run BPE on the entire corpus, the complexity would be 
O(n<sup>2</sup>) and our resulting vocabulary would likely include phrases.

We can address this issue by first tokenizing the corpus into words, and running
BPE on each word. Then, the frequency of a "byte-pair" is just the sum of all its 
word-level frequencies multiplied by the corresponding word frequency. 

For the sake of simplicity, we define words as alphabetical strings and constrain 
non-alphabetical strings to character-level representations. More sophisticated 
approaches can be used.
```python
from collections import Counter
import re

pairable_chars = "a-zA-Z"
word_counts = Counter(re.findall(f"[{pairable_chars}]+", corpus))
word_encodings = {word: [c for c in word] for word in word_counts.keys()}
```

### Build vocabulary
To run BPE, we need to run a number of iterations until our vocabulary size is 
reached or there are no more subwords. We can also speed things up by creating 
more than one "byte-pair" per iteration. 


```python
from collections import defaultdict

vocab_size = 10_000
bp_per_iter = 10
num_iter = vocab_size - len(vocab_itos)

for _ in range(num_iter):
    # generate new bytepair frequencies
    bp_counts = defaultdict(int)
    bp_words = defaultdict(set)
    for word, encodings in word_encodings.items():
        for bytepair in zip(encodings[:-1], encodings[1:]):
            bp = "".join(bytepair)
            if bp not in vocab_stoi:
                bp = " ".join(bytepair) # space to facilitate word encodings update below
                bp_counts[bp] += word_counts[word]
                bp_words[bp].add(word)

    # exit if no more subwords
    if len(bp_counts) == 0:
        break

    # update stoi/itos and word_encodings
    best_bp = sorted(bp_counts, key=bp_counts.get, reverse=True)[:bp_per_iter]
    for bp in best_bp:
        merged_bp = bp.replace(" ", "")
        vocab_stoi[merged_bp] = len(vocab_itos)
        vocab_itos += [merged_bp]
        for word in bp_words[bp]:
            word_encodings[word] = (" ".join(word_encodings[word]).replace(bp, merged_bp)).split(" ")
```

### Tokenization
With our vocabulary, we can now perform greedy subword tokenization. 
There's also a pretty interesting [paper](https://arxiv.org/abs/1804.10959) 
on the regularizing effect of tokenizing non-greedily.

```python
def tokenize(text: str) -> List[str]:
    tokens = []
    token = None
    for c in text:
        # expand previous token by one character or append previous token to tokens
        if token is not None:
            new_token = token + c
            if new_token not in vocab_stoi:
                tokens.append(token)
                token = None
            else:
                token = new_token

        # handle oov tokens
        if c not in vocab_stoi:
            tokens.append('<unk>')

        # begin new token
        elif token is None:
            token = c
    
    # append last token
    if token:
        tokens.append(token)

    return tokens

def detokenize(tokens: List[str]) -> str:
    text = "".join(tokens)
    return text

def encode(text: str) -> List[int]:
    return [vocab_stoi[s] for s in tokenize(text)]

def decode(encodings: List[int]) -> str:
    return detokenize([vocab_itos[i] for i in encodings])
```

For a more complete and customizable implementation of BPE, you can check out a 
small module I wrote here: [py-bpe](https://github.com/amr-amr/py-bpe)
