from torchtext.vocab import Glove, FastText
from vectors import FastTextOOV, Word2Vec, RandVec


def getTokenizer(tokenizer, ngrams):
    def wordTokenizer(text):
        pass

    def charTokenizer(text):
        pass

    tokenizerMap = {
        'spacy': 'spacy',
        'char': charTokenizer,
        'word': wordTokenizer
    }
    return tokenizerMap[tokenizer]


def getEmbeddings(emb, **kwargs):
    embeddings = {'Glove': Glove,
                  'FastText': FastText,
                  'FastTextOOV': FastTextOOV,
                  'word2vec': Word2Vec,
                  'rand': RandVec}

    return embeddings[emb](**kwargs)
