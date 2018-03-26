import torch
import os
import fastText
import numpy as np

from torchtext.vocab import Vectors
from torchtext.vocab import GloVe as Glv
from torchtext.vocab import FastText as Ftxt


class FastTextOOV(Vectors):
    url_base = 'https://example.com/fasttext-vectors/wiki.{}.bin'

    def __init__(self, language="en", cache=None, **kwargs):
        self.dim = 300
        url = self.url_base.format(language)
        name = os.path.basename(url)
        cache = '/scratch/madhuri/.vector_cache' if cache is None else cache

        model = os.path.join(cache, os.path.basename(name))
        self.m = fastText.FastText.load_model(model)

    def __getitem__(self, token):
        return torch.Tensor(self.m.get_word_vector(token)).view(1, -1)


class Word2Vec(Vectors):
    name_base = "{}_w2v.vectors"

    def __init__(self, language="en", **kwargs):
        self.dim = 300
        name = self.name_base.format(language)

        super(Word2Vec, self).__init__(name,
                                       cache="/scratch/madhuri/.vector_cache",
                                       **kwargs)


class RandVec(Vectors):
    def __init__(self, dim=300, **kwargs):
        self.dim = dim

    def __getitem__(self, token):
        return torch.Tensor(
                    np.random.uniform(-0.25, 0.25, self.dim)
                ).view(1, -1)


class FastText(Ftxt):
    def __init__(self, language, **kwargs):
        super(FastText, self).__init__(language=language,
                                       cache="/scratch/madhuri/.vector_cache")


class GloVe(Glv):
    def __init__(self, dim=300, name="6B", **kwargs):
        super(GloVe, self).__init__(name=name, dim=dim,
                                    cache="/scratch/madhuri/.vector_cache")
