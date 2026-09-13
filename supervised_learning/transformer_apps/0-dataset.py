#!/usr/bin/env python3
"""Class Dataset"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset():
    """ class dataset """

    def __init__(self):
        """ initialize dataset """
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
