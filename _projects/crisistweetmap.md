---
title: "CrisisTweetMap"
excerpt: "Using Natural Language Processing to categorize and map tweets in real-time during crises"
collection: projects
comments: true
---
# What
<br/><img src='https://raw.githubusercontent.com/amr-amr/CrisisTweetMap/master/doc/output.gif'>
[CrisisTweetMap](https://github.com/amr-amr/CrisisTweetMap) is a proof-of-concept I built over 24 hours as part of the 
[McHacks 7](https://mchacks7.devpost.com/) hackathon, winning first place.
It's a dynamic dashboard for visualizing crisis-related tweets in real-time.
Tweets are classified in relevant categories by a fine-tuned BERT classifier, 
and geolocated with a named-entity recognizer and Gazetteer. 
This allows tweets and their content to be plotted in real-time on a map, 
color-coded by category and recency.

# Why
During extreme events such as natural disasters or virus outbreaks, 
crisis managers are the decision makers. 
Their job is difficult since the right decision can save lives 
while the wrong decision can lead to their loss. 
Making such decisions in real-time can be daunting 
when there is insufficient information, which is often the case.

Recently, big data has gained a lot of traction in crisis management 
by addressing this issue; however it creates a new challenge. 
How can you act on data when there's just too much of it to keep up with? 
One example of this is the use of social media during crises. 
In theory, social media posts can give crisis managers 
an unprecedented level of real-time situational awareness. 
In practice, the noise-to-signal ratio and volume of social media 
is too large to be useful. 

CrisisTweetMap is meant to address this issue by visualizing social media posts 
in a way that makes them easy to parse and filter, 
so as to quickly extract useful and actionable information.


# How
## Tweet scraping
I used [Tweepy](http://docs.tweepy.org/en/latest/streaming_how_to.html#streaming-with-tweepy) 
to get live tweets from the Twitter streaming API. 
As they are scraped, tweets are immediately classified, geolocated, and pushed to a SQLite database.


## Tweet classification
I used [AllenNLP](https://allennlp.org/) to train a classifier by fine-tuning BERT,
similar to this [excellent tutorial](https://medium.com/analytics-vidhya/fine-tuning-bert-with-allennlp-7459119b736c). 
The model was trained on [labeled tweets from the MERS 2014 outbreak](https://crisisnlp.qcri.org/lrec2016/lrec2016.html) 
with the following classes:
- affected_people
- other_useful_information
- disease_transmission
- disease_signs_or_symptoms
- prevention
- treatment
- not_related_or_irrelevant
- deaths_reports

I decided to use BERT due to my limited amount of training data and the non-standard spelling in most tweets. 
These issues are addressed with BERT thanks to self-supervised pretraining and wordpiece tokenization respectively.
In the end, I obtained a [validation accuracy of 80.3%](https://github.com/amr-amr/CrisisTweetMap/blob/master/tweet_classifier/saved_models/bert_classification/metrics.json) 
with this approach. 

## Tweet geolocation
Most tweets do not contain geolocation. In these cases, I tried to approximate
a tweet's coordinates by extracting locations from the tweet text or user profile.
Initially, I tried using [spaCy's named entity recognition](https://spacy.io/usage/linguistic-features#named-entities) to extract locations.
However I switched to the more lightweight [geotext](https://readthedocs.org/projects/geotext/) 
due to the large size of the spaCy model and constraints on compute.

I then tried using [geopy](https://geopy.readthedocs.io/en/stable/) to obtain coordinates for extracted locations, and quickly ran up against rate-limits.
While I was able to alleviate this with an [lru cache](https://docs.python.org/3/library/functools.html#functools.lru_cache),
I was still having issues and decided to switch to an offline [gazetteer](https://en.wikipedia.org/wiki/Gazetteer).
Specifically, I used the [Geonames](https://www.geonames.org/) gazetteer, running on an Elasticsearch container from [mordecai](https://github.com/openeventdata/mordecai).

## Live tweet visualization
To visualize the classified and geolocated tweets, I used [Plotly Dash](https://plot.ly/dash/) to create an interactive app.
I built on top of the excellent [Dash Uber Rides Demo](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-uber-rides-demo) to
leverage [mapbox](https://www.mapbox.com/). 
The app pulls the most recent classified and geolocated tweets from the SQLite database every 3 seconds, and plots them on the map.
The color of the tweet is determined by the predicted class, while its opacity is calculated based on its recency.
