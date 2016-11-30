"""
collect.py
"""

from collections import Counter, defaultdict
from TwitterAPI import TwitterAPI
import time
import pickle
import signal
from threading import Thread, Condition
from multiprocessing import Queue
import sys
import threading

queueSize = 5000
queue = Queue(queueSize)
twitter_rate_limit = Condition()

#Filename
comments_data = "comments"
user_data = "users_friends"

# Service Tokens
consumer_key = 'uJQx8OkEByWaMaEucxyJl0t8D'
consumer_secret = 'SEbqZRSDqn37P2uGrXxWFQTLkbltsRNv2p4cZtV5eUzLJHdZcN'
access_token = '771910893881823233-KkWesZ1Li83LQcwIRDPqB7EoYI4RZGB'
access_token_secret = 'tlS3LpIG0L1JbcWVyXfSD3HB6FpWl2EOyvSL7etfCWOPL'


class ProducerThread(Thread):
    def __init__(self, run_event, service, keywords, store_comment_fnc, comments_file):
        self.run_event = run_event
        self.service = service
        self.comments = service.request('statuses/filter', {'language': 'en', 'track': keywords})
        self.store_comment_fnc = store_comment_fnc
        self.comments_file = comments_file
        Thread.__init__(self)

    def run(self):
        global queue
        for comment in self.comments:
            if not self.run_event.is_set():
                break
            try:
                print(comment['text'])
                self.store_comment_fnc(comment, self.comments_file)
            except:
                    continue
            queue.put(comment)

class ConsumerThread(Thread):

    def __init__(self, run_event, service, store_fnc, users_file):
        self.run_event = run_event
        self.service = service
        self.store_fnc = store_fnc
        self.users_file = users_file
        Thread.__init__(self)

    def run(self):
        cache = defaultdict(bool)
        global queue
        while self.run_event.is_set():
            comment = queue.get()
            user_id = comment['user']['id']
            if cache[user_id]:
                continue
            resource = "friends/ids"
            params = {'user_id': user_id}
            request = self.robust_request(self.service, resource, params)
            if request is None:
                continue
            friends = list(request)
            self.store_fnc((comment['user']['id'],friends), self.users_file)

    def robust_request(self, twitter, resource, params, max_tries=5):
        for i in range(max_tries):
            request = twitter.request(resource, params)
            if request.status_code == 200:
                return request
            else:
                print('Got error %s \nsleeping for 15 minutes.' % request.text)
                sys.stderr.flush()
                twitter_rate_limit.acquire()
                twitter_rate_limit.wait(61 * 15)
                twitter_rate_limit.release()
                if not self.run_event.is_set():
                    break


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
    run_event = threading.Event()
    run_event.set()
    service = authenticate()
    comments_file = open(comments_data, "wb")
    users_file = open(user_data, "wb")
    keywords = ['thanksgiving']
    collect_comments = ProducerThread(run_event, service, keywords, store_comment, comments_file)
    collect_friends =  ConsumerThread(run_event, service, store_comment, users_file)
    try:
        collect_comments.start()
        collect_friends.start()
    except:
        exit()

    try:
        while True:
            time.sleep(.1)
    except KeyboardInterrupt:
        print("Graceful Exit")
        run_event.clear()
        twitter_rate_limit.acquire()
        twitter_rate_limit.notify()
        twitter_rate_limit.release()
        collect_comments.join()
        collect_friends.join()
        comments_file.close()
        users_file.close()

if __name__ == '__main__':
        main()
