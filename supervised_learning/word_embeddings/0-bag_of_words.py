#!/usr/bin/env python3
"""Bag of Words embedding matrix module"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of shape (f,) containing features used
    """
    cleaned_sentences = []
    for sentence in sentences:
        text = re.sub(r'\'s\b', '', sentence)
        text = re.sub(r'[^\w\s]', '', text).lower()
        words = text.split()
        cleaned_sentences.append(words)

    if vocab is None:
        features_set = set()
        for words in cleaned_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, np.array(features)
