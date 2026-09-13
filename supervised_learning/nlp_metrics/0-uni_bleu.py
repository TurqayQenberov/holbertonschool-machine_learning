#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    parameters:
        references [list of lists]: list of reference translations
        sentence [list]: list containing the model proposed sentence

    returns:
        the unigram BLEU score
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

    # Count word frequencies in proposed sentence
    sentence_counts = {}
    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    # Calculate clipped counts
    clipped_counts = 0
    for word, count in sentence_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_count = ref.count(word)
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        clipped_counts += min(count, max_ref_count)

    precision = clipped_counts / c
    bleu_score = bp * precision

    return bleu_score
