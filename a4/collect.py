"""
collect.py
"""

from collections import Counter, defaultdict
from TwitterAPI import TwitterAPI
import time
import pickle
import signal
from threading import Thread
from multiprocessing import Queue
import sys

queueSize = 100
queue = Queue(queueSize)

#Filename
comments_data = "comments"
user_data = "users_friends"

# Service Tokens
consumer_key = 'uJQx8OkEByWaMaEucxyJl0t8D'
consumer_secret = 'SEbqZRSDqn37P2uGrXxWFQTLkbltsRNv2p4cZtV5eUzLJHdZcN'
access_token = '771910893881823233-KkWesZ1Li83LQcwIRDPqB7EoYI4RZGB'
access_token_secret = 'tlS3LpIG0L1JbcWVyXfSD3HB6FpWl2EOyvSL7etfCWOPL'


class ProducerThread(Thread):
    def __init__(self, service, keywords, store_comment_fnc, comments_file):
        self.service = service
        self.comments = service.request('statuses/filter', {'language': 'en', 'track': keywords})
        self.store_comment_fnc = store_comment_fnc
        self.comments_file = comments_file
        Thread.__init__(self)

    def run(self):
        global queue
        while True:
            for comment in self.comments:
                try:
                    print(comment['text'])
                    self.store_comment_fnc(comment, self.comments_file)
                except:
                    continue
                queue.put(comment)

class ConsumerThread(Thread):

    def __init__(self, service, store_fnc, users_file):
        self.service = service
        self.store_fnc = store_fnc
        self.users_file = users_file
        Thread.__init__(self)

    def run(self):
        cache = defaultdict(bool)
        global queue
        while True:
            comment = queue.get()
            user_id = comment['user']['id']
            if cache[user_id]:
                continue
            resource = "friends/ids"
            params = {'user_id': user_id}
            friends = list(self.robust_request(self.service, resource, params))
            self.store_fnc((comment['user']['id'],friends), self.users_file)

    def robust_request(self, twitter, resource, params, max_tries=5):
        for i in range(max_tries):
            request = twitter.request(resource, params)
            if request.status_code == 200:
                return request
            else:
                print('Got error %s \nsleeping for 15 minutes.' % request.text)
                sys.stderr.flush()
                time.sleep(61 * 15)



def authenticate():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def store_comment(comment, comments_file):
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
    users_file = open(user_data, "wb")
    keywords = ['thanksgiving']
    try:
        ProducerThread(service, keywords, store_comment, comments_file).start()
        ConsumerThread(service, store_comment, users_file).start()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        comments_file.close()
        users_file.close()


if __name__ == '__main__':
        main()
