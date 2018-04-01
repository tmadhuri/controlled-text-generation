from torchtext import data, datasets
from torchtext.vocab import GloVe, FastText

from ctextgen import utils

import random


class SST_Dataset:

    def __init__(self, emb_dim=50, mbsize=32, main=True, dataset2=None,
                 **kwargs):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, val, test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=utils.filter(6)
        )

        self.train = train

        if main:
            train_datasets = [train.text, dataset2.get_train().text] \
                             if dataset2 else [train]
            self.TEXT.build_vocab(*train_datasets, vectors=FastText('en'))
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

    def get_train(self):
        return self.train


class IMDB_Dataset:

    def __init__(self, emb_dim=50, mbsize=32, main=True, dataset2=None,
                 **kwargs):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, test = datasets.IMDB.splits(
            self.TEXT, self.LABEL, filter_pred=utils.filter(6)
        )

        self.train = train

        if main:
            train_datasets = [train.text, dataset2.get_train().text] \
                             if dataset2 else [train]
            self.TEXT.build_vocab(*train_datasets,
                                  vectors=GloVe('6B', dim=emb_dim))
            self.LABEL.build_vocab(train)

            self.n_vocab = len(self.TEXT.vocab.itos)
            print(self.n_vocab)
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

    def get_train(self):
        return self.train


class WikiText_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True, tokenize='spacy', fix_length=None)
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


class MyDataset:
    def __init__(self, dataset, emb='rand', emb_dim=300, tokenizer='word',
                 ngrams=1, mbsize=32, language='en', max_filter_size=5,
                 main=True, dataset2=None):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True,
                               tokenize=utils.getTokenizer(tokenizer, ngrams,
                                                           'en'))
        self.LABEL = data.Field(sequential=False, unk_token=None,
                                use_vocab=False)

        train, test = data.TabularDataset.splits(
            fields=[('text', self.TEXT), ('label', self.LABEL)],
            path=".data/" + dataset + "/0/", train="train.txt",
            test="test.txt", format="tsv",
            filter_pred=utils.filter(max_filter_size)
        )

        random.seed(1245)
        train, val = train.split(0.9, stratified=False, strata_field='label',
                                 random_state=random.getstate())

        self.train = train

        if main:
            train_datasets = [dataset2.get_train().text, train.text] \
                             if dataset2 else [train]
            print(train_datasets, dataset2.get_train())
            self.TEXT.build_vocab(*train_datasets,
                                  vectors=utils.getEmbeddings(emb,
                                                              dim=emb_dim,
                                                              language='en'))
            self.LABEL.build_vocab(train)

            self.n_vocab = len(self.TEXT.vocab.itos)
            print(self.n_vocab)
            self.emb_dim = emb_dim

            self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
                (train, val, test), batch_size=mbsize, device=-1, shuffle=True,
                sort_key=utils.sort_key
            )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))
        if batch.batch_size == 1:
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

    def get_train(self):
        return self.train


class MR_Dataset(MyDataset):
    def __init__(self, **kwargs):
        super(MR_Dataset, self).__init__("MR", language='en', **kwargs)


class TeSA_Dataset(MyDataset):
    def __init__(self, **kwargs):
        super(TeSA_Dataset, self).__init__("TeSA", language='te', **kwargs)


class HiSA_Dataset(MyDataset):
    def __init__(self, **kwargs):
        super(HiSA_Dataset, self).__init__("HiSA", language='hi', **kwargs)


class TrecEn_Dataset(MyDataset):
    def __init__(self, **kwargs):
        super(TrecEn_Dataset, self).__init__("TREC-En", language='en',
                                             **kwargs)


class TrecHi_Dataset(MyDataset):
    def __init__(self, **kwargs):
        super(TrecHi_Dataset, self).__init__("TREC-Hi", language='hi',
                                             **kwargs)


class MyWikiDataset:
    def __init__(self, dataset, emb='rand', emb_dim=300, tokenizer='word',
                 ngrams=1, mbsize=32, language='en', max_filter_size=5,
                 main=True, dataset2=None):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                               lower=True,
                               tokenize=utils.getTokenizer(tokenizer, ngrams,
                                                           'en'))

        train, test = data.TabularDataset.splits(
            fields=[('text', self.TEXT)],
            path=".data", train=(dataset + ".tokenized.short.out"),
            format="tsv",
            filter_pred=utils.filter(max_filter_size)
        )

        random.seed(1245)
        train, val = train.split(0.9, stratified=False, strata_field='label',
                                 random_state=random.getstate())

        self.train = train

        if main:
            train_datasets = [dataset2.get_train().text, train.text] \
                             if dataset2 else [train]
            print(train_datasets, dataset2.get_train())
            self.TEXT.build_vocab(*train_datasets,
                                  vectors=utils.getEmbeddings(emb,
                                                              dim=emb_dim,
                                                              language='en'))
            self.n_vocab = len(self.TEXT.vocab.itos)
            print(self.n_vocab)
            self.emb_dim = emb_dim

            self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
                (train, val), batch_size=mbsize, device=-1, shuffle=True,
                sort_key=utils.sort_key
            )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))
        if batch.batch_size == 1:
            batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda()

        return batch.text

    def next_validation_batch(self, gpu=False):
        batch = next(iter(self.val_iter))

        if gpu:
            return batch.text.cuda()

        return batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def get_train(self):
        return self.train


class WikiEn_Dataset(MyWikiDataset):
    def __init__(self, **kwargs):
        super(WikiEn_Dataset, self).__init__("enwiki", language='en',
                                             **kwargs)


class WikiHi_Dataset(MyWikiDataset):
    def __init__(self, **kwargs):
        super(WikiHi_Dataset, self).__init__("hiwiki", language='hi',
                                             **kwargs)


class WikiTe_Dataset(MyWikiDataset):
    def __init__(self, **kwargs):
        super(WikiTe_Dataset, self).__init__("tewiki", language='te',
                                             **kwargs)
