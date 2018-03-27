import numpy as np
import cPickle
from collections import defaultdict
import sys
import re
import pandas as pd
import pickle
from fastText_multilingual.fasttext import FastVector
from fastText import FastText
from libindic.syllabifier import Syllabifier
from hyphen import Hyphenator

CHAR = 0
dim = 300
SYL = 1


def build_data_cv(dataset, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """

    if dataset[:2].lower() == "mr":
        data_folder = ["rt-polarity.pos", "rt-polarity.neg"]
        if dataset == "mr_te":
            data_folder = ["mr_te.pos", "mr_te.neg"]
        if dataset == "mr_hi":
            data_folder = ["mr_hi.pos", "mr_hi.neg"]

        revs = []
        pos_file = data_folder[0]
        neg_file = data_folder[1]
        vocab = defaultdict(float)

        l = [0, 0]

        with open(pos_file, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.decode("utf8", "ignore").strip().lower())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()
                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    en = Hyphenator('en_US')
                    syl_split = map(lambda x: en.syllables(x)
                                    if (len(x) > 1 and len(en.syllables(x)) > 0)
                                    else [x],
                                    words)
                    syl_split = map(lambda x: x[:-1] + [x[-1] + u">"],
                                    map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1
                datum = {"y": 1,
                         "text": words,
                         "num_words": len(words),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
                l[1] += 1
        with open(neg_file, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.decode("utf8", "ignore").strip().lower())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()
                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    en = Hyphenator('en_US')
                    syl_split = map(lambda x: en.syllables(x)
                                    if (len(x) > 1 and len(en.syllables(x)) > 0)
                                    else [x],
                                    words)
                    syl_split = map(lambda x: x[:-1] + [x[-1] + u">"],
                                    map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1
                datum = {"y": 0,
                         "text": words,
                         "num_words": len(words),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
                l[0] += 1
        print l
        return revs, vocab

    elif dataset[-4:] == "TeSA":
        revs = []
        data_file = "ACTSA_telugu_polarity_annotated_UTF.txt"
        if dataset != "TeSA":
            data_file = dataset + ".txt"

        l = [0, 0, 0]

        vocab = defaultdict(float)
        with open(data_file, "rb") as f:
            for line in f:
                line = line.decode('utf-8', 'ignore').strip().split(" ", 1)
                label = int(line[0].strip())
                if label == 0:
                    label = 2
                if label == -1:
                    label = 0
                if label != 0 and label != 1 and label != 2:
                    print label

                l[label] += 1

                rev = []
                rev.append(line[1].strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()
                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    te = Syllabifier()
                    syl_split = map(lambda x: te.syllabify_te(x)
                                    if (len(x) > 1 and len(te.syllabify_te(x)) > 0)
                                    else [x],
                                    words)
                    syl_split = map(lambda x: x[:-1] + [x[-1] + u">"], map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1
                datum = {"y": label,
                         "text": words,
                         "num_words": len(words),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
        print l
        return revs, vocab

    elif dataset[-4:] == "HiSA":
        revs = []
        data_file = "5001-end.txt"
        if dataset != "HiSA":
            data_file = dataset + ".txt"

        l = [0, 0, 0]

        vocab = defaultdict(float)
        with open(data_file, "rb") as f:
            for line in f:
                line = line.decode('utf-8', 'ignore').strip().split("\t", 1)
                if((len(line) < 2) or ((line[1].strip() != 'p') and
                   (line[1].strip() != 'n') and (line[1].strip() != 'o'))):
                    continue

                label = 2
                if line[1].strip() == 'p':
                    label = 1
                elif line[1].strip() == 'n':
                    label = 0

                l[label] += 1

                rev = []
                rev.append(line[0].strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()
                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    hi = Syllabifier()
                    syl_split = map(lambda x: hi.syllabify_hi(x)
                                    if (len(x) > 1 and len(hi.syllabify_hi(x)) > 0)
                                    else [x],
                                    words)
                    syl_split = map(lambda x: x[:-1] + [x[-1] + u">"],
                                    map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1
                datum = {"y": label,
                         "text": words,
                         "num_words": len(words),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)

        print l
        return revs, vocab

    elif dataset[:4] == "TREC":
        revs = []
        data_file = []
        if dataset[-2:] == "En":
            data_file = ["train_5500.label", "TREC_10.label"]
        elif dataset[-2:] == "Hi":
            data_file = ["train_5500.hi.label", "TREC_10.hi.label"]
        elif dataset[-3:] == "w2w":
            data_file = ["train_5500.hi_w2w_en.label",
                         "TREC_10.hi_w2w_en.label"]

        train_file = data_file[0]
        test_file = data_file[1]

        l = [0, 0, 0, 0, 0, 0]

        classes = {"DESC": 0, "ENTY": 1, "ABBR": 2, "HUM": 3, "NUM": 4,
                   "LOC": 5}

        vocab = defaultdict(float)
        with open(train_file, "rb") as f:
            for line in f:
                line = line.decode('utf-8').strip().split(" ", 1)

                if((len(line) < 2) or (line[0].split(":")[0] not in classes)):
                    continue

                label = classes[line[0].split(":")[0]]

                l[label] += 1

                rev = []
                rev.append(line[1].strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()
                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    if dataset[-2:] == "En":
                        en = Hyphenator('en_US')
                        syl_split = map(lambda x: en.syllables(x)
                                        if (len(x) > 1 and len(en.syllables(x)) > 0)
                                        else [x],
                                        words)
                        syl_split = map(lambda x: x[:-1] + [x[-1] + u">"],
                                        map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    else:
                        hi = Syllabifier()
                        syl_split = map(lambda x: hi.syllabify_hi(x)
                                        if (len(x) > 1 and len(hi.syllabify_hi(x)) > 0)
                                        else [x],
                                        words)
                        syl_split = map(lambda x: x[:-1] + [x[-1] + u">"],
                                        map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1

                datum = {"y": label,
                         "text": words,
                         "num_words": len(words),
                         "split": 1}
                revs.append(datum)

        t = [0, 0, 0, 0, 0, 0]
        with open(test_file, "rb") as f:
            for line in f:
                line = line.decode('utf-8').strip().split(" ", 1)

                if((len(line) < 2) or (line[0].split(":")[0] not in classes)):
                    continue

                label = classes[line[0].split(":")[0]]

                t[label] += 1

                rev = []
                rev.append(line[1].strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev), dataset)
                else:
                    orig_rev = " ".join(rev).lower()

                words = orig_rev.split()
                if CHAR != 0 and SYL == 0:
                    words = reduce(lambda y, z: y+z,
                                   map(lambda x: [x[i:i+CHAR]
                                                  for i in
                                                  range(max(1,
                                                            len(x) - CHAR + 1
                                                            )
                                                        )
                                                  ],
                                       words))
                elif SYL != 0:
                    if dataset[-2:] == "En":
                        en = Hyphenator('en_US')
                        syl_split = map(lambda x: en.syllables(x)
                                        if (len(x) > 1 and len(en.syllables(x)) > 0)
                                        else [x],
                                        words)
                        syl_split = map(lambda x: x[:-1] + [x[-1] + u">"], map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    else:
                        hi = Syllabifier()
                        syl_split = map(lambda x: hi.syllabify_hi(x)
                                        if (len(x) > 1 and len(hi.syllabify_hi(x)) > 0)
                                        else [x],
                                        words)
                        syl_split = map(lambda x: x[:-1] + [x[-1] + u">"], map(lambda x: [u"<" + x[0]] + x[1:], syl_split))
                    comb_syl_split = map(lambda x: ["".join(x[i:i + SYL])
                                                    for i in
                                                    range(max(len(x) - SYL + 1,
                                                              1))
                                                    ], syl_split)
                    words = reduce(lambda x, y: x + y, comb_syl_split)
                for word in set(words):
                    vocab[word] += 1

                datum = {"y": label,
                         "text": words,
                         "num_words": len(words),
                         "split": 0}
                revs.append(datum)

        print l, t
        return revs, vocab


def get_W(word_vecs, k=dim):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word.decode('utf-8', 'ignore').strip() in vocab:
                word_vecs[word.decode('utf-8', 'ignore').strip()] \
                    = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def load_word_vectors(fname, vocab):
    words, vecs = pickle.load(open(fname, "rb"))
    word_vecs = {}

    for i in range(len(words)):
        if words[i] in vocab:
            word_vecs[words[i]] = vecs[i]

    return word_vecs


def load_dssm_vectors(fname, vocab, lang):
    data = pickle.load(open(fname, "rb"))
    word_vecs = {}

    W = []
    idx_map = {}

    if lang == "en":
        W = data["doc_emb_wts"]
        idx_map = data["doc_word_idx_map"]
    else:
        W = data["source_emb_wts"]
        idx_map = data["source_word_idx_map"]

    for word in vocab:
        if word in idx_map:
            word_vecs[word] = W[idx_map[word]]

    return word_vecs


def load_fastText_vectors(fname, model, vocab):
    vectorFile = open(fname, "r")

    m = FastText.load_model(model)

    word_vecs = {}

    for line in vectorFile:
        line = line.split()

        if len(line) != (dim + 1):
            continue

        if line[0].decode('utf8').strip() in vocab:
            word_vecs[line[0].decode('utf8').strip()] = np.array(
                                                                line[1:],
                                                                dtype="float32"
                                                                )

    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = m.get_word_vector(word)

    return word_vecs


def load_fastText_vectors_char(model, vocab):
    m = FastText.load_model(model)

    word_vecs = {}

    for subword in vocab:
        if subword not in word_vecs:
            word_vecs[subword] = m.get_word_vector(subword)

    # for subword in vocab:
    #     if subword not in word_vecs:
    #         index = m.get_subword_id(subword)
    #         word_vecs[subword] = m.get_input_vector(index)

    return word_vecs


def load_fastText_trans_vectors(fname, vocab, lang):
    word_vecs = {}

    words = FastVector(vector_file=fname)
    words.apply_transform("fastText_multilingual/alignment_matrices/"
                          + lang + ".txt")

    for word in words.id2word:
        if word.decode('utf8').strip() in vocab:
            word_vecs[word.decode('utf8').strip()] = words[word]

    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=dim):
    """
    For words that occur in at least min_df documents,
    create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as
    pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, dataset="MR", TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    if dataset == "MR" or dataset == "mr":
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    dataset = sys.argv[1]
    vecType = sys.argv[2]

    print "loading data...",
    revs, vocab = build_data_cv(dataset, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"

    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)

    W = []
    rand_vecs = {}
    W2 = []
    word_idx_map = {}

    if vecType == "w2v":
        w2v_file = "../../datasets/GoogleNews-vectors-negative300.bin"

        if dataset == "MR" or dataset[:2] == "En" or dataset[:2] == "en" \
                or dataset[-2:] == "En" or dataset[-3:] == "w2w":
            w2v_file = "../../datasets/w2v/en_w2v.vectors"
        elif dataset == "TeSA" or dataset[:2] == "Te" or dataset[:2] == "te" \
                or dataset[-2:] == "Te":
            w2v_file = "../../datasets/w2v/te_w2v.vectors"
        elif dataset == "HiSA" or dataset[:2] == "Hi" or dataset[:2] == "hi" \
                or dataset[-2:] == "Hi":
            w2v_file = "../../datasets/w2v/hi_w2v.vectors"

        print "loading word2vec vectors...",
        w2v = load_bin_vec(w2v_file, vocab)
        print "word2vec loaded!"
        print "num words already in word2vec: " + str(len(w2v))
        add_unknown_words(w2v, vocab)
        W2, word_idx_map = get_W(w2v)

    elif vecType == "word_emb":
        word_emb_file = ""

        if dataset == "MR" or dataset[:2] == "En" or dataset[:2] == "en" \
                or dataset[-2:] == "En" or dataset[-3:] == "w2w":
            word_emb_file = "../../datasets/polyglot/polyglot-en.pkl"
        elif dataset == "TeSA" or dataset[:2] == "Te" or dataset[:2] == "te" \
                or dataset[-2:] == "Te":
            word_emb_file = "../../datasets/polyglot/polyglot-te.pkl"
        elif dataset == "HiSA" or dataset[:2] == "Hi" or dataset[:2] == "hi" \
                or dataset[-2:] == "Hi":
            word_emb_file = "../../datasets/polyglot/polyglot-hi.pkl"

        print "loading word vectors...",
        word_vecs = load_word_vectors(word_emb_file, vocab)
        print "word vectors loaded!"
        print "num words already in word vectors: " + str(len(word_vecs))
        add_unknown_words(word_vecs, vocab)
        W2, word_idx_map = get_W(word_vecs)

    elif vecType == "fastText":
        word_emb_file = ""
        model = ""

        if dataset == "MR" or dataset[:2] == "En" or dataset[:2] == "en" \
                or dataset[-2:] == "En" or dataset[-3:] == "w2w":
            word_emb_file = "../../datasets/fastTextVectors/wiki.en.vec"
            model = "../../datasets/fastTextVectors/wiki.en.bin"
        elif dataset == "TeSA" or dataset[:2] == "Te" or dataset[:2] == "te" \
                or dataset[-2:] == "Te":
            word_emb_file = "../../datasets/fastTextVectors/wiki.te.vec"
            model = "../../datasets/fastTextVectors/wiki.te.bin"
        elif dataset == "HiSA" or dataset[:2] == "Hi" or dataset[:2] == "hi" \
                or dataset[-2:] == "Hi":
            word_emb_file = "../../datasets/fastTextVectors/wiki.hi.vec"
            model = "../../datasets/fastTextVectors/wiki.hi.bin"

        print "loading word vectors...",
        word_vecs = load_fastText_vectors(word_emb_file, model, vocab)
        print "word vectors loaded!"
        print "num words already in word vectors: " + str(len(word_vecs))
        add_unknown_words(word_vecs, vocab)
        W2, word_idx_map = get_W(word_vecs)

    elif vecType == "fastTextChar":
        word_emb_file = ""
        model = ""

        if dataset == "MR" or dataset[:2] == "En" or dataset[:2] == "en" \
                or dataset[-2:] == "En" or dataset[-3:] == "w2w":
            model = "../../datasets/fastTextVectors/wiki.en.bin"
        elif dataset == "TeSA" or dataset[:2] == "Te" or dataset[:2] == "te" \
                or dataset[-2:] == "Te":
            model = "../../datasets/fastTextVectors/wiki.te.bin"
        elif dataset == "HiSA" or dataset[:2] == "Hi" or dataset[:2] == "hi" \
                or dataset[-2:] == "Hi":
            model = "../../datasets/fastTextVectors/wiki.hi.bin"

        print "loading word vectors...",
        word_vecs = load_fastText_vectors_char(model, vocab)
        print "word vectors loaded!"
        print "num words already in word vectors: " + str(len(word_vecs))
        add_unknown_words(word_vecs, vocab)
        W2, word_idx_map = get_W(word_vecs)

    elif vecType == "fastTextTrans":
        word_emb_file = ""
        lang = ""

        if dataset == "MR" or dataset[:2] == "En" or dataset[:2] == "en" \
                or dataset[-2:] == "En":
            word_emb_file = "../../datasets/fastTextVectors/wiki.en.vec"
            model = "../../datasets/fastTextVectors/wiki.en.bin"
            lang = "en"

        elif dataset == "TeSA" or dataset[:2] == "Te" or dataset[:2] == "te" \
                or dataset[-2:] == "Te":
            word_emb_file = "../../datasets/fastTextVectors/wiki.te.vec"
            model = "../../datasets/fastTextVectors/wiki.te.bin"
            lang = "te"

        elif dataset == "HiSA" or dataset[:2] == "Hi" or dataset[:2] == "hi" \
                or dataset[-2:] == "Hi":
            word_emb_file = "../../datasets/fastTextVectors/wiki.hi.vec"
            model = "../../datasets/fastTextVectors/wiki.hi.bin"
            lang = "hi"

        print "loading word vectors...",
        # word_vecs = load_fastText_trans_vectors(word_emb_file, vocab, lang)
        word_vecs = load_fastText_vectors(word_emb_file, model, vocab)
        print "word vectors loaded!"
        print "num words already in word vectors: " + str(len(word_vecs))
        add_unknown_words(word_vecs, vocab)
        W2, word_idx_map = get_W(word_vecs)

        transmat = np.loadtxt("fastText_multilingual/alignment_matrices/"
                              + lang + ".txt")
        W2 = np.matmul(W2, transmat)

    elif vecType == "dssm":
        dssm_char_size = sys.argv[3]
        dssm_filter_size = sys.argv[4]
        dssm_units = sys.argv[5]
        dssm_dim = sys.argv[6]
        dssm_word_dim = sys.argv[7]
        dssm_model = sys.argv[8]
        dssm_lang = sys.argv[9]

        if dssm_model != "same":
            dssm_model = ""
        else:
            dssm_model = ".same"

        lang = ""
        if dataset == "HiSA" or dataset == "TREC-Hi":
            lang = "hin"
        elif dataset == "TeSA":
            lang = "tel"
        else:
            lang = "en"

        word_emb_file = "dssm_" + dssm_dim + "_" + dssm_units \
                        + "_[" + dssm_filter_size \
                        + "]_" + dssm_word_dim + "_" + dssm_char_size \
                        + "." + dssm_lang + dssm_model + ".pickle"

        print "loading word vectors...",
        word_vecs = load_dssm_vectors(word_emb_file, vocab, lang)
        print "word vectors loaded!"
        print "num words already in word vectors: " + str(len(word_vecs))
        add_unknown_words(word_vecs, vocab)
        W2, word_idx_map = get_W(word_vecs)

        dataset = dataset.lower() + dssm_model + "_" + dssm_lang \
            + "_" + dssm_word_dim + "_" + dssm_filter_size + "_" \
            + dssm_units

    elif vecType == "oneHot":
        W2 = np.concatenate((np.zeros(shape=(1, len(vocab))),
                             np.identity(len(vocab))),
                            axis=0)

    else:
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab)
        W2, word_idx_map = get_W(rand_vecs)

    if SYL != 0:
        cPickle.dump([revs, W2, word_idx_map, vocab, max_l],
                     open(dataset.lower() + "_" + str(SYL) + "_" +
                          str(dim) + ".p",
                          "wb"))
    else:
        cPickle.dump([revs, W2, word_idx_map, vocab, max_l],
                     open(dataset.lower() + "_" + str(CHAR) + "_" +
                          str(dim) + ".p",
                          "wb"))
    print "dataset created!"
