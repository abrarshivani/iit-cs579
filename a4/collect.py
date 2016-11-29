"""
collect.py
"""

from collections import Counter, defaultdict
from TwitterAPI import TwitterAPI
import time
import pickle
import signal

#Filename
comments_data = "comments"

# Service Tokens
consumer_key = 'uJQx8OkEByWaMaEucxyJl0t8D'
consumer_secret = 'SEbqZRSDqn37P2uGrXxWFQTLkbltsRNv2p4cZtV5eUzLJHdZcN'
access_token = '771910893881823233-KkWesZ1Li83LQcwIRDPqB7EoYI4RZGB'
access_token_secret = 'tlS3LpIG0L1JbcWVyXfSD3HB6FpWl2EOyvSL7etfCWOPL'


def authenticate():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def store_comment(comment, comments_file):
    print(comment['text'])
    pickle.dump(comment, comments_file)


def get_comments(service, keywords, store_comment_fnc, comments_file):
    comments = service.request('statuses/filter', {'language': 'en', 'track': keywords})
    for comment in comments:
        try:
            store_comment_fnc(comment, comments_file)
        except:
            pass


def main():
    service = authenticate()
    comments_file = open(comments_data, "wb")
    keywords = ['thanksgiving']
    try:
        get_comments(service, keywords, store_comment, comments_file)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        comments_file.close()

if __name__ == '__main__':
        main()
