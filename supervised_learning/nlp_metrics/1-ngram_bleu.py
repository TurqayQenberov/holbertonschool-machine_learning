#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score for a sentence
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    parameters:
        references [list of lists]: list of reference translations
        sentence [list]: list containing the model proposed sentence
        n [int]: size of the n-gram to use for evaluation

    returns:
        the n-gram BLEU score
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

    # Generate n-grams for the sentence
    sentence_ngrams = [
        tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)
    ]
    total_sentence_ngrams = len(sentence_ngrams)

    if total_sentence_ngrams == 0:
        return 0.0

    # Count sentence n-grams
    sentence_ngram_counts = {}
    for ngram in sentence_ngrams:
        sentence_ngram_counts[ngram] = (
            sentence_ngram_counts.get(ngram, 0) + 1
        )

    # Calculate clipped counts across all references
    clipped_counts = 0
    for ngram, count in sentence_ngram_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_ngrams = [
                tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
            ]
            ref_count = ref_ngrams.count(ngram)
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        clipped_counts += min(count, max_ref_count)

    precision = clipped_counts / total_sentence_ngrams
    bleu_score = bp * precision

    return bleu_score
