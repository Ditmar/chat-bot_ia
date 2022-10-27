import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
spanishStemmer=SnowballStemmer("spanish", ignore_stopwords=True)

"""TOKENIZACION"""
def tokenize(sentence):
    """
    Aqui se dividira la oracion en una matriz de palabras
    o tokens ya sea una palabra, caracter, puntuacion o
    un numero
    """
    return nltk.word_tokenize(sentence)

"""STEMMING"""
def stem(word):
    """
    Aqui se encontrara la palabra raiz de cada palabra
    encontrada en la matriz
    """
    return spanishStemmer.stem(word.lower())


"""Bag Of Words"""
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

















