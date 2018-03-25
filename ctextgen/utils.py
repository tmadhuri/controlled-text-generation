from ctextgen.vectors import FastTextOOV, Word2Vec, RandVec, GloVe, FastText


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
    embeddings = {'Glove': GloVe,
                  'FastText': FastText,
                  'FastTextOOV': FastTextOOV,
                  'word2vec': Word2Vec,
                  'rand': RandVec}

    return embeddings[emb](**kwargs)


def sort_key(ex):
    return len(ex.text)
