import nltk_utils
import numpy as np
import pytest

@pytest.mark.parametrize(
    "input_a, input_b, output", [
        (["hola", "como", "estas", "tu"], 
        ["hey", "hola", "yo", "tu", "adios", "gracias", "genial"],
        [0, 1, 0, 1, 0, 0, 0]),
        (["informacion", "vemos", "direccion", "tu"],
        ["hey", "hola", "yo", "tu", "adios", "gracias", "genial"],
        [0, 0, 0, 1, 0, 0, 0])
    ]
)
def test(input_a, input_b, output):
    tokenized_sentence = np.array(input_a)
    words = np.array(input_b)
    new_words = [nltk_utils.stem(word) for word in words] 
    result = np.array(output)
    total = np.array(nltk_utils.bag_of_words(tokenized_sentence, new_words))
    np.testing.assert_equal(total, result)