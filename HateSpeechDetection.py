#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tweepy

# initialize api instance
consumer_key = '5bZAf2lKwTBu6dVZONz5tSZji'
consumer_secret='eDVqN7vZbEni0Ho4dLc5drP6vMdwPJIHrk4WRGcYL0MyV9YzG0'
access_token = '893669907304292353-8ZgFZNnBemESKRiaBb9lHyRPN3LV7a8'
access_token_secret = 'UoL2ojUPP2q0mYvUeKCqFTmGs9bn8TsTlUSFtolLyeKhB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

import csv
csvFile = open('result.csv.txt','a')
csvWriter = csv.writer(csvFile)

api = tweepy.API(auth, wait_on_rate_limit=True)
for tweet in tweepy.Cursor(api.search, q="jihadism", lang="en").items(100):
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

csvFile.close()



import codecs
import nltk
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

with codecs.open('result.csv.txt','r') as csvfile:
    for tweet in csvfile:

        tokenized_tweets = tknzr.tokenize(tweet)

        with open('result1.csv.txt', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(tokenized_tweets)

filename = 'result1.csv.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
# import os
# os.remove('result.csv')
# os.remove('result1.csv')
import re
alpha_num_values = re.split(r'\W+', text)


filtered_tokens = [x for x in alpha_num_values if not any(c.isdigit() for c in x)]
#print(filtered_tokens)

lemma = nltk.wordnet.WordNetLemmatizer()
stemmed_words = [lemma.lemmatize(word) for word in filtered_tokens]
# print(stemmed_words)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in stemmed_words if not w in stop_words]
# print(words)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()
for word in words:
  vs = analyzer.polarity_scores(word)

  # print("{:-<65} {}".format(word, str(vs)))


    # Create a SentimentIntensityAnalyzer object.
sid_obj = SentimentIntensityAnalyzer()
negative_list = []


for word in words:
    sentiment_dict = sid_obj.polarity_scores(word)

    if sentiment_dict['compound'] <= - 0.05:
        # print("Overall sentiment dictionary is : ", sentiment_dict)
        #
        #
        # print("word was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        negative_list.append(word)
# print(negative_list)

#
# from sklearn.feature_extraction.text import TfidfVectorizer
# # create the transform
# vectorizer = TfidfVectorizer()
# # tokenize and build vocab
# vectorizer.fit(negative_list)
# # summarize
# print(vectorizer.vocabulary_)
# #print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform(negative_list)
#
# # summarize encoded vector
# # print(vector.shape)
#
# # print(vector.toarray)
# vectors = vectorizer.idf_

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
import nltk

from sklearn import cluster
from sklearn import metrics

model = Word2Vec(negative_list, min_count=1)

# get vector data
X = model[model.wv.vocab]
# print(X)
#
# print(model.similarity('this', 'is'))
#
# print(model.similarity('post', 'book'))
#
# print(model.most_similar(negative=[], topn=2))
#
# print(model['the'])

# print(list(model.vocab))
#
# print(len(list(model.vocab)))

NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    print(word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)


# In[ ]:




