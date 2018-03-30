import sys
import cPickle
import numpy as np
import random

from keras.models import Model
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, LSTM
from keras.layers import Dropout, Embedding, Input, BatchNormalization, Bidirectional
from keras.layers.merge import Concatenate
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical


def train_cnn(datasets, params, maxl, U, dataset, char_size):
    model_input = Input(shape=(maxl + 2 * (params["filter_size"][-1] - 1), ))
    z = Embedding(U.shape[0], params["dim"],
                  input_length=maxl + 2 * (params["filter_size"][-1] - 1),
                  name="embeddings")(model_input)

    conv_blocks = []
    for sz in params["filter_size"]:
        conv = Conv1D(filters=params["units"],
                      kernel_size=sz,
                      activation=params["activation"], name="conv_" + str(sz))(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(params["dropout"])(z)
    model_output = Dense(len(params["labels"]), activation="softmax", name="dense1")(z)

    model = Model(model_input, model_output)


    model.compile(loss='categorical_crossentropy',
                  optimizer="adadelta", metrics=["accuracy"])

    print "Compiled"


    # model.load_weights("cnn.h5", by_name=True)


    embedding_layer = model.get_layer("embeddings")
    embedding_layer.set_weights([U])

    model.fit(datasets["x_train"], to_categorical(datasets["y_train"]),
              validation_split=0.05, epochs=params["epochs"], verbose=2,
              shuffle=True, batch_size=50)

    # model.save_weights("cnn.h5")

    # print "model saved"

    y_pred = model.predict(datasets["x_test"])
    print y_pred


    # cPickle.dump({"y_test": datasets["y_test"], "y_pred": np.argmax(y_pred, axis=1)},
    #              open(str(dataset) + "_" + str(char_size) + "_" \
    #                   + str(params["units"]) + "_" + str(params["dim"]) + "_" \
    #                   + str(params["filter_size"][0]) + ".pickle", "wb"))

    return accuracy_score(datasets["y_test"], np.argmax(y_pred, axis=1))


def get_idx_from_sent(sent, word_idx_map, max_l, filter_size):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_size - 1
    words = sent
    for i in xrange(pad):
        x.append(0)

    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    while len(x) < (max_l + 2 * pad):
        x.append(0)

    return np.array(x, dtype="float32")


def make_idx_data_cv(revs, word_idx_map, cv, max_l, filter_size):
    x_train, x_test = [], []
    y_train, y_test = [], []

    random.shuffle(revs)

    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_size)

        if rev["split"] == cv:
            x_test.append(sent)
            y_test.append(rev["y"])
        else:
            x_train.append(sent)
            y_train.append(rev["y"])

    print len(x_train), len(x_test)

    x_train = np.array(x_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")
    x_test = np.array(x_test, dtype="float32")
    y_test = np.array(y_test, dtype="float32")

    return {"x_train": x_train, "y_train": y_train,
            "x_test": x_test, "y_test": y_test}


if __name__ == "__main__":
    print "loading data...",

    dataset = sys.argv[1]
    char_size = sys.argv[2]
    filter_size = sys.argv[3]
    units = sys.argv[4]
    dim = sys.argv[5]

    labels = []

    if dataset[:2] == "MR":
        labels = [0, 1]
    elif dataset[-4:] == "TeSA" or dataset[-4:] == "HiSA":
        labels = [0, 1, 2]
    elif dataset[:4] == "TREC":
        labels = [0, 1, 2, 3, 4, 5]

    x = cPickle.load(open(dataset.lower() + "_" + char_size + "_" + dim + ".p",
                          "rb"))
    revs, W2, word_idx_map, vocab, maxl = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"

    print "model architecture: CNN-non-static"
    non_static = True
    print "using: random vectors"
    U = W2

    cv = 10
    if dataset[:4] == "TREC":
        cv = 1

    results = []
    r = range(0, cv)
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, maxl,
                                    int(filter_size))

        params = {"units": int(units),
                  "activation": "relu",
                  "bias": True,
                  "dropout": 0.5,
                  "labels": labels,
                  "batch_size": 50,
                  "epochs": 25,
                  "dim": int(dim),
                  "filter_size": [int(filter_size)]}

        perf = train_cnn(datasets, params, maxl, U, dataset, char_size)

        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)

    print str(np.mean(results))
