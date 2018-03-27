from torchtext import data, datasets
from torchtext.vocab import GloVe, FastText

from ctextgen import utils

import random


class SST_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

        train, val, test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=FastText('en'))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1, shuffle=True
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(iter(self.val_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]


class IMDB_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15

        train, test = datasets.IMDB.splits(
            self.TEXT, self.LABEL, filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, _ = data.BucketIterator.splits(
            (train, test), batch_size=mbsize, device=-1, shuffle=True
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])


class WikiText_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, val, test = datasets.WikiText2.splits(self.TEXT)

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, _, _ = data.BPTTIterator.splits(
            (train, val, test), batch_size=10, bptt_len=15, device=-1
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))
        return batch.text.cuda() if gpu else batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])


class MR_Dataset:
    def __init__(self, emb='rand', emb_dim=300,
                 tokenizer='spacy', ngrams=1,
                 mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True,
                               tokenize=utils.getTokenizer(tokenizer, ngrams))
        self.LABEL = data.Field(sequential=False, unk_token=None,
                                use_vocab=False)

        train, test = data.TabularDataset.splits(
            fields=[('text', self.TEXT), ('label', self.LABEL)],
            path=".data/MR/0/", train="train.txt",
            test="test.txt", format="tsv"
        )

        random.seed(1245)
        train, val = train.split(0.9, stratified=False, strata_field='label', random_state=random.getstate())

        self.TEXT.build_vocab(train,
                              vectors=utils.getEmbeddings(emb,
                                                          dim=emb_dim,
                                                          language='en'))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1, shuffle=True,
            sort_key=utils.sort_key
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(iter(self.val_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]
