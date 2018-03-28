import torch

from ctextgen.dataset import SST_Dataset
from ctextgen.dataset import MR_Dataset, TeSA_Dataset, HiSA_Dataset
from ctextgen.dataset import TrecEn_Dataset, TrecHi_Dataset
from ctextgen.model import RNN_VAE
from ctextgen import utils

import argparse
import time


parser = argparse.ArgumentParser(
    description='Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--model', default='ctextgen', metavar='',
                    help="choose the model: {`vae`, `ctextgen`}, \
                          (default: `ctextgen`)")

datasets = {
    'sst': SST_Dataset,
    'mr': MR_Dataset,
    'tesa': TeSA_Dataset,
    'hisa': HiSA_Dataset,
    'trec-en': TrecEn_Dataset,
    'trec-hi': TrecHi_Dataset
}

parser.add_argument('--dataset', type=lambda d: datasets[d.lower()],
                    choices=datasets.values(), required=True,
                    help='Dataset to be used.')

parser.add_argument('-t', '--tokenizer', type=str,
                    choices=["char", "word", "syl", "spacy"], required=True,
                    help='Tokenizer to use')

parser.add_argument('-n', '--ngrams', type=int, default=1,
                    help='Size of ngrams')

parser.add_argument('-e', '--embeddings', type=str,
                    choices=['Glove', 'word2vec', 'FastText', 'FastTextOOV'],
                    default='rand',
                    help='Which embeddings to use.')

parser.add_argument('-d', '--dimension', type=int, default=300,
                    help='Size of embedding vector')

parser.add_argument('--freeze_emb', default=False, action='store_true',
                    help='Whether to freeze embeddings while training.')

parser.add_argument('-c', '--num_classes', type=int, default=3,
                    help='Number of classes in dataset.')

parser.add_argument('-f', '--filters', type=int, default=3,
                    nargs='+',
                    help='Filters for the discriminator.')

parser.add_argument('-u', '--units', type=int, default=100,
                    help='Number of filters in discriminator.')

args = parser.parse_args()

dataset = args.dataset(tokenizer=args.tokenizer,
                       ngrams=args.ngrams,
                       emb=args.embeddings,
                       emb_dim=args.dimension,
                       max_filter_size=max(args.filters))


mb_size = 50
z_dim = 20
h_dim = args.dimension
lr = 1e-3
lr_decay_every = 1000000
n_iter = 10000
log_interval = 1000
z_dim = h_dim
c_dim = args.num_classes

torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=args.freeze_emb, gpu=args.gpu
)

if args.gpu:
    model.load_state_dict(torch.load('models/{}.bin'
                                     .format(args.model
                                             + utils.getModelName(args))))
else:
    model.load_state_dict(torch.load('models/{}.bin'
                                     .format(args.model
                                             + utils.getModelName(args)),
                                     map_location=lambda storage, loc: storage)
                          )

outputFile = open("gen" + utils.getModelName(args) + ".out.txt", "w+")

for i in range(n_iter):
    # Samples latent and conditional codes randomly from prior
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)

    # Generate positive sample given z
    for j in range(args.num_classes):
        for k in range(args.num_classes):
            c[0, k] = 0

        c[0, j] = 1

        _, c_idx = torch.max(c, dim=1)
        sample_idxs = model.sample_sentence(z, c, temp=0.1)

        outputFile.write('{}\t{}\n'.format(dataset.idxs2sentence(sample_idxs)
                                           .encode('utf8'),
                                           str(dataset.idx2label(int(c_idx)))))


outputFile.close()
