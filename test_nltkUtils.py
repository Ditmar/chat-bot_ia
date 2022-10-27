import nltk_utils
import numpy as np
import pytest


@pytest.mark.parametrize(
    "sentence, result", [
        ("¿como estas tu?",
        ["como", "estas", "tu"])
    ]
)
def test_NLTK(sentence, result):
    ignore_words = '¿?,.¡!'
    res = sentence.translate(str.maketrans('', '', ignore_words))
    sentence_tokenize = nltk_utils.tokenize(res)
    sentence_stemming = [nltk_utils.stem(w) for w in sentence_tokenize] 
    assert (sentence_stemming == result )



