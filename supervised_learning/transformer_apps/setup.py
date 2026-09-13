#!/usr/bin/env python3
"""setup: isolates the tensorflow_datasets dependency behind load_pt2en.

The rest of the project is only allowed to import `tensorflow`,
`transformers`, and `load_pt2en` from this module. Everything needed
to actually pull down the raw Portuguese/English sentence pairs lives
here instead.
"""
import tensorflow_datasets as tfds


def load_pt2en():
    """Load the ted_hrlr_translate/pt_to_en corpus.

    Returns:
        data_train: tf.data.Dataset of (pt, en) raw string tensor pairs
        data_valid: tf.data.Dataset of (pt, en) raw string tensor pairs
    """
    examples, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                            with_info=True,
                            as_supervised=True)

    data_train, data_valid = examples['train'], examples['validation']

    return data_train, data_valid
