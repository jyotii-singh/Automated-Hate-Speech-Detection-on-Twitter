# Automated-Hate-Speech-Detection-on-Twitter
This Project detects top words that promote hate speech in tweets regarding sensitive topics using natural language processing techniques.

In this work, to identify the tweets that generate hate speech from the tweets fetched for a
particular keyword,we studied an approach in which after preprocessing of tweets fetched
,sentiment analysis is applied using VADER tool to obtain the list of negative words.This
list is then converted into vector by using a Term Frequency-Inverse Document
Frequency (TF-IDF) vectorizer that measures how important a word is.K-means
clustering which is last step and yet to perform will be applied on the resulting matrix
obtained after vectorization to determine the most discussed topics.
