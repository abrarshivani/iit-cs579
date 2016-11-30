"""
classify.py
"""

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
import string
import tarfile
import urllib.request
import pickle
from nltk.corpus import sentiwordnet as swn


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'],
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'],
          dtype='<U5')
    """
    doc = doc.lower()
    punctuation = re.sub("_", "", string.punctuation)
    if not keep_internal_punct:
        doc = re.sub("\W+", " ", doc)
    doc = doc.split()
    return np.array([term.rstrip(punctuation).lstrip(punctuation) for term in doc if term.rstrip(punctuation) != ""])

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    if feats is None:
        return
    prefix = "token="
    for token in tokens:
        feats[prefix+token] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    windows = []
    prefix_token_pair = "token_pair="
    token_delimiter = "__"
    if k <= 0 or feats is None:
        return
    for window in range(0,len(tokens)- k + 1):
        windows.append(tokens[window: window + k])
    for window in windows:
        for combination in combinations(window, 2):
            feat = prefix_token_pair + combination[0] + token_delimiter + combination[1]
            feats[feat] += 1

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    pos_words_key = "pos_words"
    neg_words_key = "neg_words"
    if feats is None:
        return
    feats[pos_words_key] = 0
    feats[neg_words_key] = 0
    for token in tokens:
        token = token.lower()
        if token in pos_words:
            feats[pos_words_key] += 1
        elif token in neg_words:
            feats[neg_words_key] += 1


def token_lexicon_features(tokens, feats):
    cache = defaultdict(list)
    if feats is None:
        return
    for token in tokens:
        polarity = []
        token = token.lower()
        if len(cache[token]) == 0:
            polarity = list(swn.senti_synsets(token))
            if len(polarity) != 0:
                cache[token] = [polarity[0].pos_score(), polarity[0].neg_score()]
            else:
                cache[token] = [0, 0]
        if (cache[token][0] > cache[token][1]) or (token in pos_words):
            feats[token] += (cache[token][0] * 100)
        elif (cache[token][0] < cache[token][1]) or (token in neg_words):
            feats[token] += (cache[token][1] * 100)

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    for feature_fn in feature_fns:
        feature_fn(tokens, feats)
    return sorted(feats.items())


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    doc_feature_counter = defaultdict(lambda: 0)
    features_list = []
    global_feature_list = []
    prune_features_list = []

    data = []
    row = []
    col = []

    for tokens in chain(tokens_list):
        features_list.append(featurize(tokens, feature_fns))

    if vocab is None:
        for features in features_list:
            for feature in features:
                doc_feature_counter[feature[0]] += 1

        for features in chain(features_list):
            prune_feature = []
            for feature in features:
                if doc_feature_counter[feature[0]] >= min_freq:
                    prune_feature.append(feature)
            prune_features_list.append(prune_feature)
            global_feature_list += prune_feature

        index = 0
        vocab = defaultdict(int)
        for feature in sorted(chain(global_feature_list), key = lambda feature_with_count: feature_with_count[0]):
            if vocab.get(feature[0]) is None:
                vocab[feature[0]] = index
                index += 1

    col_size = len(vocab)
    for row_no, features in enumerate(features_list):
        for feature in chain(features):
            if vocab.get(feature[0]) is None:
                continue
            data.append(feature[1])
            row.append(row_no)
            col.append(vocab[feature[0]])
    X = csr_matrix((data, (row, col)), shape=(len(tokens_list),col_size),dtype=int)
    return X,vocab


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    results = []
    all_feature_fns = []
    for level in range(1, len(feature_fns)+1):
        for combination in combinations(feature_fns, level):
            all_feature_fns.append(list(combination))
    for punct_val in punct_vals:
        tokens_list = [tokenize(d, punct_val) for d in docs]
        for min_freq in min_freqs:
            for feature_fn in all_feature_fns:
                clf = svm.LinearSVC()
                X,vocab = vectorize(tokens_list, feature_fn, min_freq)
                accuracy = cross_validation_accuracy(clf, X, labels, 5)
                result = {}
                result['features'] = tuple(feature_fn)
                result['punct'] = punct_val
                result['accuracy'] = accuracy
                result['min_freq'] = min_freq
                results.append(result)
   # print(sorted(results, key=lambda result: result['accuracy'], reverse=True))
    return sorted(chain(results), key=lambda result: result['accuracy'], reverse=True)


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    if results == None or len(results) == 0:
        return
    filename = "accuracies.png"
    accuracies = sorted(list(map(lambda result: result['accuracy'], results)))
    plt.plot(np.arange(len(accuracies)), accuracies)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig(filename)


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    mean = []
    accuracies_per_setting = defaultdict(list)
    if len(results) == 0:
        return mean
    for result in results:
        for setting in result.keys():
            if setting == "accuracy":
                continue
            if setting == "features":
                setting_value = ""
                for value in result[setting]:
                    setting_value += " " + value.__name__
                setting_value = setting_value.lstrip()
            else:
                setting_value = result[setting]
            setting_parameter_value = setting + "=" + str(setting_value)
            accuracies_per_setting[setting_parameter_value].append(result["accuracy"])
    for setting, accuracies in accuracies_per_setting.items():
        mean.append((np.mean(accuracies), setting))
    return sorted(chain(mean), reverse=True)

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    vocab = {}
    clf = svm.LinearSVC()
    if best_result is None or docs is None or labels is None:
        return clf, vocab
    tokens_list = [tokenize(d, best_result["punct"]) for d in docs]
    X,vocab = vectorize(tokens_list, best_result["features"], best_result["min_freq"])
    clf.fit(X, labels)
    return clf, vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    results = []
    if clf is None or label is None or n is None or vocab is None:
        return results
    if n <= 0 or len(vocab) == 0 or len(clf.coef_) == 0:
        return results
    results = zip(sorted(vocab, key=vocab.get), clf.coef_[0])
    if label == 0:
        results = [(result[0], result[1]*-1) for result in results if result[1] < 0]
    else:
        results = [result for result in results if result[1] >= 0]
    return sorted(chain(results), key=lambda result: -result[1])[:n]


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    docs, labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(d, best_result["punct"]) for d in docs]
    X,vocab = vectorize(tokens_list, best_result["features"], best_result["min_freq"], vocab)
    return docs, labels, X


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    misclassified_docs = []
    T = clf.predict_proba(X_test)
    predictions = clf.predict(X_test)
    for index_of_doc in range(0,len(test_docs)):
        test_class = predictions[index_of_doc]
        if test_class != test_labels[index_of_doc]:
            misclassified_docs.append((test_docs[index_of_doc], T[index_of_doc][test_class], test_class))
    misclassified_docs = sorted(chain(misclassified_docs), key=lambda misclassified_doc: -misclassified_doc[1])[:n]
    for misclassified_doc in misclassified_docs:
        print("truth=%d predicted=%d proba=%f" % ((1 - misclassified_doc[2]), misclassified_doc[2], misclassified_doc[1]))
        print(misclassified_doc[0])

def read_real_data(filename, db, best_result, vocab):
    data = load_pickle_file(filename)
    docs = get_docs_from_data(data)
    preprocessed_docs = preprocess_comments(docs, db)
    tokens_list = [tokenize(d, best_result["punct"]) for d in preprocessed_docs]
    X,vocab = vectorize(tokens_list, best_result["features"], best_result["min_freq"], vocab)
    return preprocessed_docs, X

def predict_real_data(docs, X, clf):
    results = []
    predictions = clf.predict(X)
    for index_of_doc in range(0,len(docs)):
        label = predictions[index_of_doc]
        results.append((docs[index_of_doc], label))
    return results

def get_docs_from_data(data):
    return [comments['text'] for comments in data]

def load_pickle_file(filename):
    with open(filename, "rb") as handle:
         while True:
            try:
                 yield pickle.load(handle)
            except EOFError:
                break

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

def remove_url(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def preprocess_comment(comment, db):
    #comment = comment.encode('ascii', 'ignore')
    comment = comment.lower()
    comment = remove_url(comment)
    comment = replace_hashtag_word(comment)
    comment = remove_usernames(comment)
    comment = strip_punctuation(comment)
    comment = replace_dict_def(comment, db)
    comment = remove_non_word_chars(comment)
    comment = remove_extra_whitespaces(comment)
    return comment


def preprocess_comments(comments, db):
    results = []
    for comment in comments:
        results.append(preprocess_comment(comment, db))
    return results


def train():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    #print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    #print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    return best_result, vocab, clf


def get_summary(results):
    number_of_instances_per_class_found = defaultdict(int)
    class_examples = {}
    for result in results:
        number_of_instances_per_class_found[result[1]] += 1
        class_examples[result[1]] = result[0]
    return number_of_instances_per_class_found, class_examples


def write_summary(filename, results):
    number_of_instances_per_class_found, class_examples = get_summary(results)
    with open(filename, "w") as handle:
        handle.write("Number of instances per class found: ")
        for label, instances in number_of_instances_per_class_found.items():
            handle.write(str(instances) + " ")
        handle.write("\nOne example from each class: \n")
        for label, examples in class_examples.items():
            handle.write("Class %s:- %s\n" % (label, examples))


def main():
    filename = "comments"
    slang_dict = build_slang_dict("slang.txt")
    best_result, vocab, clf = train()
    docs, X = read_real_data(filename, slang_dict, best_result, vocab)
    results = predict_real_data(docs, X, clf)
    write_summary("classify_summary", results)

if __name__ == '__main__':
    main()
