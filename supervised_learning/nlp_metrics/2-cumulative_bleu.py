#!/usr/bin/env python3
"""
Calculates the cumulative n-gram BLEU score for a sentence
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    parameters:
        references [list of lists]: list of reference translations
        sentence [list]: list containing the model proposed sentence
        n [int]: size of the largest n-gram to use for evaluation

    returns:
        the cumulative n-gram BLEU score
    """
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]

    # Best match length (r)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # Brevity Penalty (BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - (r / c))

    precisions = []

    # Calculate modified precision for each n-gram level from 1 to n
    for i in range(1, n + 1):
        sentence_ngrams = [
            tuple(sentence[j:j + i]) for j in range(len(sentence) - i + 1)
        ]
        total_sentence_ngrams = len(sentence_ngrams)

        if total_sentence_ngrams == 0:
            precisions.append(0)
            continue

        sentence_ngram_counts = {}
        for ngram in sentence_ngrams:
            sentence_ngram_counts[ngram] = (
                sentence_ngram_counts.get(ngram, 0) + 1
            )

        clipped_counts = 0
        for ngram, count in sentence_ngram_counts.items():
            max_ref_count = 0
            for ref in references:
                ref_ngrams = [
                    tuple(ref[k:k + i]) for k in range(len(ref) - i + 1)
                ]
                ref_count = ref_ngrams.count(ngram)
                if ref_count > max_ref_count:
                    max_ref_count = ref_count
            clipped_counts += min(count, max_ref_count)

        precision = clipped_counts / total_sentence_ngrams
        precisions.append(precision)

    precisions = np.array(precisions)

    if np.any(precisions == 0):
        return 0.0

    # Geometric mean with equal weights (1 / n)
    weights = np.ones(n) / n
    geometric_mean = np.exp(np.sum(weights * np.log(precisions)))

    bleu_score = bp * geometric_mean

    return bleu_score
