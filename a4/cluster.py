"""
cluster.py
"""

import re
import string

filename = "comments"


def build_slang_dict(filename):
    slang = {}
    lines = []
    with open(filename, "r") as handle:
        lines = handle.readlines()
    for line in lines:
        words = line.split("\n")[0].split("  -   ")
        slang[words[0]] = words[1]
    return slang

def remove_extra_whitespaces(text):
    text = re.sub('[\s]+', ' ', text)
    return text

def gethashtags(text):
    hashtag_regex = '#([^\s]+)'
    hashtags = re.findall(hashtag_regex, text)
    return hashtags


def replace_hashtag_word(text):
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def remove_usernames(text):
    text = re.sub('@[^\s]+', ' ', text)
    return text


def remove_non_word_chars(text):
    text = re.sub('[^A-Za-z\s]*', '', text)
    return text


def replace_dict_def(text, db):
    words = text.split(" ")
    text = ""
    for word in words:
        try:
            meaning = db[word]
            text = text + meaning + " "
        except:
            text = text + word + " "
    return text


def strip_punctuation(tweet):
    tweet = "".join(c for c in tweet if c not in string.punctuation)
    return tweet


def preprocess_comment(comment, db):
    #comment = comment.encode('ascii', 'ignore')
    comment = replace_hashtag_word(comment)
    comment = remove_usernames(comment)
    comment = strip_punctuation(comment)
    comment = replace_dict_def(comment, db)
    comment = remove_non_word_chars(comment)
    comment = remove_extra_whitespaces(comment)
    return comment


def preprocess_comments(comments, db):
    result = None
    for comment in comments:
        comment = preprocess_comment(comment, db)
    return comment


def main():
    comments = ["This is lol"]
    slang_dict = build_slang_dict("slang.txt")
    #comments = get_comments_from_file()
    preprocessed_comments = preprocess_comments(comments, slang_dict)
    print(preprocessed_comments)
    #classified_comments = classify_comments(preprocessed_comments)
    #store_results(classified_comments)


if __name__ == '__main__':
    main()
