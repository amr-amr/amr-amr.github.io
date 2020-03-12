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

### Scraping submissions
First, let's instantiate a generator for submissions (posts) on a given subreddit,
starting from a given date. We also specify which fields we want to keep in a 
submission object to reduce bandwidth.

```python
from psaw import PushshiftAPI
import datetime as dt


submission_filter = [
    'author',
    'author_fullname',
    'full_link',
    'is_self',
    'num_comments',
    'score',
    'selftext',
    'title',
    'id',
]
start_dt = int(dt.datetime(2019, 1, 1).timestamp())
api = PushshiftAPI()
posts_gen = api.search_submissions(
    after=start_dt,
    subreddit="amitheasshole",
    filter=submission_filter
)
```

Next, we start generating some submissions! I decided to simply write them to a
dataframe which I save every 25,000th submission (in case something crashes).

```python
import pandas as pd

df_posts = pd.DataFrame()
posts = []
for post in posts_gen:
    posts.append(post.d_)
    if len(posts) == 25_000:
        df_posts = df_posts.append(pd.DataFrame(posts))
        df_posts.to_pickle("aita_2019_posts.pkl")
        posts = []
```

### Scraping comments
Now let's scrape comments. Similarly to the submission generator, we define 
a start date and filter. However, we also define a query `q` to only retrieve comments
containing a specific string (in our case, the labels we're interested in).
Here, I decided to only scrape 100,000 comments for each label, 
since I was short on time.

```python
df_comments = pd.DataFrame()
for q in ['NTA', 'YTA', 'ESH', 'NAH', 'INFO']:
    comments_gen = api.search_comments(
        after=start_dt,
        subreddit='amitheasshole',
        filter=comment_filter,
        q=q,
    )
    comments = []
    for comment in comments_gen:
        comments.append(comment.d_)
        if len(comments) == 100_000:
            df_comments = df_comments.append(pd.DataFrame(comments))
            df_comments.to_pickle('aita_2019_comments.pkl')
            break
    df_comments.to_pickle('aita_2019_comments.pkl')
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
