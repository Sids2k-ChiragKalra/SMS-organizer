import operator
import re

import numpy as np

import data.normalise as nm

hp_frequency_word = 4
hp_similarity = 0.75
hp_word_stemming = False


def get_words(messages):
    words = {}

    # store number of times each word appears
    for message in messages:
        for word in message.split():
            # if len(word) > 2 and not re.search(r'\d', word):
            word = word.strip()
            words[word] = words.get(word, 0) + 1

    # sort words in descending order
    words = dict(sorted(words.items(), key=operator.itemgetter(1), reverse=True))

    selected = []
    for word in words:
        if words[word] >= hp_frequency_word:
            selected.append(word)
        else:
            break

    return selected


def similar_features(a, b):
    n = min(np.sum(a), np.sum(b)) + 1
    allowed_mismatch = int(n*(1-hp_similarity))
    count = np.sum(np.logical_xor(a, b))
    return count <= allowed_mismatch


def compute_features(messages, compute=False):
    messages = np.vectorize(nm.stem)(messages)

    selected = get_words(messages) if compute else \
        np.genfromtxt("data/pruned_db/words.csv", delimiter=',', encoding='utf8', dtype=str)

    features = []

    for message in messages:
        feature = []
        for pw in selected:
            feature.append(int(pw in message))
        features.append(feature)

    return np.array(selected), np.array(features)
