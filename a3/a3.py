# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def flatten_list(lists):
    return [value for sublist in lists for value in sublist]

def tfidf(docs):
    """
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      docs...The list of docs
    Returns:
        [[(term1, tfidf),(term2, tfidf)], [(term1, tfidf)]]
    """
    tfidf_data = []
    df = defaultdict(lambda: 0)
    N = len(docs)

    #Calculate df[i]
    for doc in docs:
        for token in set(doc):
            df[token] += 1
    for doc in docs:
        term_freqs = Counter(doc)
        k = term_freqs.most_common(1)
        tfidf_value_of_terms_in_doc = []
        for term, freq in term_freqs.items():
            if len(k) == 0 or len(k[0]) != 2:
                break
            if df[term] == 0:
                df[term] = 1
            tf = freq / k[0][1]
            idf = math.log10(N / df[term])
            tfidf_value = tf * idf
            tfidf_value_of_terms_in_doc.append((term, tfidf_value))
        tfidf_data.append(tfidf_value_of_terms_in_doc)
    return tfidf_data

def dot(a, b):
    return a.multiply(b).sum()

def norm(matrix):
    return math.sqrt(dot(matrix, matrix))

def create_csr_matrix(tfidf_value_per_doc, vocab):
    col = []
    row = []
    data = []
    row_no = 0

    for tfidf_value_of_term in tfidf_value_per_doc:
        data.append(tfidf_value_of_term[1])
        row.append(row_no)
        col.append(vocab[tfidf_value_of_term[0]])
    X = csr_matrix((data, (row, col)), shape=(1, len(vocab)),dtype=float)
    return X

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    tokens = []
    tokensColumnName = "tokens"
    #if movies empty, append a new column, datatype of new column?
    #if movies not empty
    for genre in movies['genres']:
        tokens.append([tokenize_string(genre)])
    movies = movies.join(pd.DataFrame(tokens, columns=[tokensColumnName]))
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    features = []
    featuresColumnName = "features"
    vocab = defaultdict(lambda: len(vocab))
    tokens = sorted(list(set(flatten_list(movies['tokens'].tolist()))))
    for token in tokens:
        vocab[token]
    tfidf_values = tfidf(movies['tokens'].tolist())
    for tfidf_value in tfidf_values:
        features.append([create_csr_matrix(tfidf_value, vocab)])
    movies = movies.join(pd.DataFrame(features, columns=[featuresColumnName]))
    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    cosine_similarity = 0.0
    if a == None or b == None:
        return cosine_similarity
    num = dot(a, b)
    deno = norm(a) * norm(b)
    return num / deno

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    predictions = []
    for test in ratings_test.iterrows():
        prediction = 0.0
        wts = []
        num = []
        ratings_of_user = []
        test_movie_feature = movies[movies.movieId == test[1].movieId].iloc[0].features
        for train in ratings_train[ratings_train.userId == test[1].userId].iterrows():
            if test[1].movieId == train[1].movieId:
                continue
            train_movie_feature = movies[movies.movieId == train[1].movieId].iloc[0].features
            sim = cosine_sim(train_movie_feature, test_movie_feature)
            ratings_of_user.append(train[1].rating)
            if sim > 0:
                num.append(sim * train[1].rating)
                wts.append(sim)
        if len(wts) == 0:
            prediction = np.average(ratings_of_user)
        else:
            prediction = sum(num)/sum(wts)
        predictions.append(prediction)
    return np.array(predictions)

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
