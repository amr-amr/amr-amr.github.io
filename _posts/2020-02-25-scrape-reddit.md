---
title: 'How to scrape reddit to create your own dataset'
date: 2020-02-25
permalink: /posts/2020/02/scrape-reddit/
tags:
  - nlp
  - code
---

In this post, I'll go over the basics of byte-pair encoding (BPE), outline its 
advantages as a tokenization algorithm in natural language processing, 
and show you some code.

{% include toc %}

## What is BPE
[Byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) 
is a simple data compression algorithm that recursively combines most 
frequently co-occurring atoms (byte-pairs) into new atoms:

```python
# encoded string        atoms
s = 'aaabdaaabacabaa'   # {a,b,c,d}
s = 'ZabdZabacabZ'      # {Z=aa,a,b,c,d}
s = 'ZYdZYacYZ'         # {Z=aa,Y=ab,a,b,c,d}
s = 'XdXacYZ'           # {Z=aa,Y=ab,X=ZY,a,b,c,d}
```

With minor modifications, BPE can be used on a corpus of natural 
language text to create a set of atoms that contains frequent words and subwords, 
as well as characters. 

This subword-level representation has many advantages for NLP tasks, which is why 
it has been successfully used in many recent state-of-the-art language 
representation models such as BERT and GPT-2.

## Advantages of BPE
Converting text to a format that allows it to be input into machine learning 
models is an important part of NLP. This typically involves tokenization:
splitting the text into tokens that can be mapped to a vocabulary. These 
can then be converted to numerical representations such as word embeddings.

Generally, tokenization was done on a word-level basis. However, this leads to 
the issue of __out-of-vocabulary__ words, whereby new words can not be represented.
New words can include misspellings, rare words such as
["Penrhyndeudraeth"](https://en.wikipedia.org/wiki/Penrhyndeudraeth), or 
[neologisms](https://en.wikipedia.org/wiki/Neologism) such as 
["yeeted"](https://www.urbandictionary.com/define.php?term=Yeet). 
Character-level models can address this, but presumably have limited representational 
capacities compared to word-level models, since words are very much more than 
the sum of their characters. 

Subword-level models represent the best of both worlds. They address the issue 
of out-of-vocabulary words while maintaining rich word-level representations, 
and can potentially learn relevant morphological subword representations. For example, 
if "yeet" and "-ing" are in my vocabulary but I have never seen "yeeting", I can
still infer that "yeeting" means "to yeet".

## BPE in code
### Initial vocabulary
First we load our corpus and define our initial vocabulary as all the 
[latin unicode characters](https://en.wikipedia.org/wiki/Plane_(Unicode)#Basic_Multilingual_Plane) 
and any other characters in our corpus.
However, its worth noting that many models use actual bytes to support all 
languages with a smaller vocabulary. 
```python
from pathlib import Path

vocab_itos = ['<unk>'] + [chr(i) for i in range(0x0000, 0x024f)]
vocab_stoi = {s: i for i,s in enumerate(vocab_itos)}

corpus = Path("corpus.txt").read_text()
for c in set(corpus):
    if c not in vocab_stoi:
        vocab_stoi[c] = len(vocab_itos)
        vocab_itos += [c]
```

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
