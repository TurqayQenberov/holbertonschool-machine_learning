#!/usr/bin/env python3
"""Class Dataset"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset():
    """Class Dataset"""

    def __init__(self):
        """Class constructor"""
        self.data_train, self.data_valid = load_pt2en()

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """tokenize data """
        pt_base = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True)
        en_base = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True)

        def pt_sentences():
            """iterator"""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_sentences():
            """iterator"""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_sentences(), vocab_size=2 ** 15)
        tokenizer_en = en_base.train_new_from_iterator(
            en_sentences(), vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encoding """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_vocab = self.tokenizer_pt.vocab_size
        en_vocab = self.tokenizer_en.vocab_size

        lang1 = [pt_vocab] + self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False) + [pt_vocab + 1]

        lang2 = [en_vocab] + self.tokenizer_en.encode(
            en_text, add_special_tokens=False) + [en_vocab + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        """ tf_encode"""
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
