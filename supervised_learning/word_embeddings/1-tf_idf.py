#!/usr/bin/env python3
"""TF-IDF embedding matrix module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of shape (f,) containing features used
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
