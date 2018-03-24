import torch
import os
import fastText
import numpy as np

from torchtext.vocab import Vectors


class FastTextOOV(Vectors):
    url_base = 'https://example.com/fasttext-vectors/wiki.{}.bin'

    def __init__(self, language="en", cache=None, **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        cache = '.vector_cache' if cache is None else cache

        model = os.path.join(cache, os.path.basename(name))
        self.m = fastText.FastText.load_model(model)

    def __getitem__(self, token):
        return torch.Tensor(self.m.get_word_vector(token)).view(1, -1)


class Word2Vec(Vectors):
    name_base = "{}_w2v.vectors"

    def __init__(self, language="en", **kwargs):
        name = self.name_base.format(language)

        super(Word2Vec, self).__init__(name, **kwargs)


class RandVec(Vectors):
    def __init__(self, dim=300, **kwargs):
        self.dim = dim

    def __getitem__(self, token):
        return torch.Tensor(
                    np.random.uniform(-0.25, 0.25, self.dim)
                ).view(1, -1)
