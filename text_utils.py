# IPA Phonemizer: https://github.com/bootphon/phonemizer

import os
import os.path as osp
import pandas as pd

_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_number = '0123456789'


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_number) + list("'・-()？")

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes
