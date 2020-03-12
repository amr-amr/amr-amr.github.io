---
title: 'How to scrape reddit to create your own dataset'
date: 2020-02-25
permalink: /posts/2020/02/scrape-reddit/
tags:
  - nlp
  - code
---

{% include toc %}

A lot of subreddits have specific submission/comment formats which can make
for interesting NLP datasets. In this post, I'll show you how to create your own
dataset from [reddit.com/r/AmITheAsshole](https://www.reddit.com/r/AmITheAsshole).


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


## Which library to use
I had previously used [praw](https://github.com/praw-dev/praw) to scrape reddit 
live, however this option requires credentials to use reddit’s API and the 
rate-limits can be restrictive if you’re trying to scrape a large historic dataset.

Instead, I opted to use [psaw](https://github.com/dmarx/psaw) which wraps 
the pushshift.io API and is much more forgiving for scraping larger historic datasets.  

## Code
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

Now let's scrape comments. Similarly to the submission generator, we define 
a start date and filter. However, we also define a query `q` to only retrieve comments
containing a specific string (in our case, the labels we're interested in).
Here, I decided to only scrape 100,000 comments for each label, 
since I was short on time.

```python
comment_filter = [
    'author',
    'author_fullname',
    'body',
    'is_submitter',
    'id',
    'link_id', # post id
    'parent_id', # parent id = link id when top level comment
    'score',
    'total_awards_received',
]
df_comments = pd.DataFrame()
for q in ["NTA", "YTA", "ESH", "NAH", "INFO"]:
    comments_gen = api.search_comments(
        after=start_dt,
        subreddit="amitheasshole",
        filter=comment_filter,
        q=q,
    )
    comments = []
    for comment in comments_gen:
        comments.append(comment.d_)
        if len(comments) == 100_000:
            df_comments = df_comments.append(pd.DataFrame(comments))
            df_comments.to_pickle("aita_2019_comments.pkl")
            break
    df_comments.to_pickle("aita_2019_comments.pkl")
```

Now that we have our submissions and comments, we need to clean them.
First, we remove posts which do not have the right prefix and are thus not likely 
to be an "Am I The Asshole?" question. We also remove posts which do not have 
descriptions (selftext). Lastly, we remove posts which have no comments.

```python

def clean_posts(df):
    df = df.loc[(df["title"].str.startswith("AITA")) | (df["title"].str.startswith("WIBTA"))]
    df = df.loc[~(df["selftext"] == "[removed]")]
    df = df.loc[~(pd.isna(df["selftext"]))]
    df = df.loc[~df.selftext == ""]
    df = df.loc[df["num_comments"] > 0]
    return df
```

The process for cleaning comments is a little more involved. 
First, we keep only top-level comments which are a direct response to the submission, and
not to another comment. Then we remove comments from submissions we did not scrape. 
Lastly, we remove comments which have no labels or more than one.

```python
def clean_comments(df, post_ids):
    df = df.loc[df["parent_id"] == df["link_id"]]
    df["link_id"] = df["link_id"].apply(lambda x: x[3:])
    df = df.loc[df["link_id"].isin(post_ids)]

    def find_labels(text: str):
        return [q for q in ["NTA", "YTA", "ESH", "NAH", "INFO"] if q in text]

    df["labels"] = df["body"].apply(lambda x: find_labels(x))
    df["num_labels"] = df["labels"].apply(lambda x: len(x))
    df = df.loc[df["num_labels"] == 1]
    df["labels"] = df["labels"].apply(lambda x: x[0])
    return df
```

Lastly, we can use the `clean_posts` and `clean_comments` functions to merge 
together submissions and comments and get the class probabilities for each post.
```python
def merge_comments_and_posts(df_posts, df_comments):
    # map labels to indices
    itol = ["NTA", "YTA", "ESH", "NAH", "INFO"]
    ltoi = {l:i for i,l in enumerate(itol)}

    # clean posts and comments
    df_posts = clean_posts(df_posts)
    post_ids = df_posts.id.to_list()
    df_comments = clean_comments(df_comments, post_ids)
    
    # get label scores/counts for each post
    comment_labels = df_comments.labels.to_list()
    comment_post_ids = df_comments.link_id.to_list()
    comment_score = df_comments.score.to_list()
    post_labels_dict = {post_id: [0,0,0,0,0] for post_id in post_ids}
    for post_id, label, score in zip(comment_post_ids, comment_labels, comment_score):
        post_labels_dict[post_id][ltoi[label]] += score
    df_posts["label_counts"] = [post_labels_dict[post_id] for post_id in post_ids]
    
    # get label probabilities for each post
    df_posts["label_sum"] = df_posts["label_counts"].apply(lambda x: sum(x))
    df_posts = df_posts[df_posts["label_sum"] > 0]
    df_posts["label_probs"] = [[c/s for c in counts] for counts, s in zip(
        df_posts["label_counts"], df_posts["label_sum"])]

    df_posts.to_pickle("aita_2019_posts_labeled.pkl")
    df_comments.to_pickle("aita_2019_comments_cleaned.pkl")
```
That's it! You can find the whole code [here](https://github.com/amr-amr/am-i-the-asshole/blob/master/data/get_data.py).
