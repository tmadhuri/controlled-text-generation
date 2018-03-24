import argparse
import cPickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save train-test splits for CVs.")
    parser.add_argument("dataset", help="Name of the dataset.")
    args = parser.parse_args()

    os.mkdir(args.dataset)

    cv = 10
    if args.dataset[:4] == "TREC":
        cv = 1

    data = cPickle.load(open(os.path.join("rawDataFiles",
                                          (args.dataset + "_0_300").lower()
                                          + ".p"),
                             "rb"))[0]

    for i in range(cv):
        os.mkdir(os.path.join(args.dataset, str(i)))

        train = open(os.path.join(args.dataset, str(i), "train.txt"),
                     "w+")
        test = open(os.path.join(args.dataset, str(i), "test.txt"),
                    "w+")

        for sent in data:
            if sent["split"] == i:
                test.write(sent["orig_text"].encode("utf8") + "\t" +
                           str(sent["y"]) + "\n")

            else:
                train.write(sent["orig_text"].encode("utf8") + "\t" +
                            str(sent["y"]) + "\n")

        train.close()
        test.close()
