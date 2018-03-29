from ctextgen.vectors import FastTextOOV, Word2Vec, RandVec, GloVe, FastText

from libindic.syllabifier import Syllabifier
from hyphen import Hyphenator
from functools import reduce


def getTokenizer(tokenizer, ngrams, language):
    def max_len(x):
        return max(1, len(x) - ngrams + 1)

    def wordTokenizer(text):
        return text.split()

    def charTokenizer(text):
        words = wordTokenizer(text)
        char_ngrams = map(lambda x: [x[i:i+ngrams] for i in range(max_len(x))],
                          words)
        return reduce(lambda y, z: y+z, char_ngrams)

    def sylTokenizer(text):
        words = wordTokenizer(text)

        if language == 'en':
            en = Hyphenator('en_US')
            syl_split = map(lambda x: en.syllables(x)
                            if (len(x) > 1 and len(en.syllables(x)) > 0)
                            else [x],
                            words)
            comb_syl_split = map(lambda x: ["".join(x[i:i + ngrams])
                                            for i in
                                            range(max(len(x) - ngrams + 1,
                                                      1))
                                            ], syl_split)
            return reduce(lambda x, y: x + y, comb_syl_split)
        elif language == 'te':
            te = Syllabifier()
            syl_split = map(lambda x: te.syllabify_te(x)
                            if (len(x) > 1 and len(te.syllabify_te(x)) > 0)
                            else [x],
                            words)
            comb_syl_split = map(lambda x: ["".join(x[i:i + ngrams])
                                            for i in
                                            range(max(len(x) - ngrams + 1,
                                                      1))
                                            ], syl_split)
            return reduce(lambda x, y: x + y, comb_syl_split)

        else:
            hi = Syllabifier()
            syl_split = map(lambda x: hi.syllabify_hi(x)
                            if (len(x) > 1 and len(hi.syllabify_hi(x)) > 0)
                            else [x],
                            words)
            comb_syl_split = map(lambda x: ["".join(x[i:i + ngrams])
                                            for i in
                                            range(max(len(x) - ngrams + 1,
                                                      1))
                                            ], syl_split)
            return reduce(lambda x, y: x + y, comb_syl_split)

    tokenizerMap = {
        'spacy': 'spacy',
        'char': charTokenizer,
        'word': wordTokenizer,
        'syl': sylTokenizer
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


def filter(max_filter_size):
    def filter_pred(ex):
        return ((len(ex.text) > max_filter_size) and (len(ex.text) <= 15))

    return filter_pred


def getModelName(params):
    modelName = "".join(["_", params.dataset.__name__,
                         "_t", params.tokenizer,
                         "_n", str(params.ngrams),
                         "_e", params.embeddings,
                         "_d", str(params.dimension),
                         "_f", str(params.filters),
                         "_u", str(params.units)])

    if params.freeze_emb:
        modelName += "_freeze"

    return modelName
