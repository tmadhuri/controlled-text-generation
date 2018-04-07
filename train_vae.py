import os
import torch
import torch.optim as optim

from ctextgen.dataset import SST_Dataset, IMDB_Dataset
from ctextgen.dataset import MR_Dataset, TeSA_Dataset, HiSA_Dataset
from ctextgen.dataset import TrecEn_Dataset, TrecHi_Dataset
from ctextgen.dataset import WikiEn_Dataset, WikiHi_Dataset, WikiTe_Dataset
from ctextgen.model import RNN_VAE
from ctextgen import utils

import argparse


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, \
                 with c ~ p(c)'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--use_saved', default=False, action='store_true',
                    help='whether to use saved model or not')

datasets = {
    'sst': SST_Dataset,
    'mr': MR_Dataset,
    'tesa': TeSA_Dataset,
    'hisa': HiSA_Dataset,
    'trec-en': TrecEn_Dataset,
    'trec-hi': TrecHi_Dataset,
    'imdb': IMDB_Dataset,
    'wikien': WikiEn_Dataset,
    'wikihi': WikiHi_Dataset,
    'wikite': WikiTe_Dataset
}

parser.add_argument('dataset', type=lambda d: datasets[d.lower()],
                    choices=datasets.values(),
                    help='Dataset to be used.')

parser.add_argument('-d2', '--dataset2', type=lambda d: datasets[d.lower()],
                    choices=datasets.values(), required=True,
                    help='2nd Dataset to be used.')

parser.add_argument('-t', '--tokenizer', type=str,
                    choices=["char", "word", "syl", "spacy"], required=True,
                    help='Tokenizer to use')

parser.add_argument('-n', '--ngrams', type=int, default=1,
                    help='Size of ngrams')

parser.add_argument('-e', '--embeddings', type=str,
                    choices=['Glove', 'word2vec', 'FastText', 'FastTextOOV',
                             'rand'],
                    default='rand',
                    help='Which embeddings to use.')

parser.add_argument('-d', '--dimension', type=int, default=300,
                    help='Size of embedding vector')

parser.add_argument('--freeze_emb', default=False, action='store_true',
                    help='Whether to freeze embeddings while training.')

parser.add_argument('-c', '--num_classes', type=int, default=3,
                    help='Number of Classes in Dataset.')

parser.add_argument('-f', '--filters', type=int, default=3,
                    nargs='+',
                    help='Filters for the discriminator.')

parser.add_argument('-u', '--units', type=int, default=100,
                    help='Number of filters in discriminator.')

parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Number of filters in discriminator.')


args = parser.parse_args()


dataset2 = args.dataset2(tokenizer=args.tokenizer,
                         ngrams=args.ngrams,
                         emb=args.embeddings,
                         emb_dim=args.dimension,
                         max_filter_size=max(args.filters),
                         main=False,
                         mbsize=args.batch_size)

dataset = args.dataset(tokenizer=args.tokenizer,
                       ngrams=args.ngrams,
                       emb=args.embeddings,
                       emb_dim=args.dimension,
                       max_filter_size=max(args.filters),
                       dataset2=dataset2,
                       mbsize=args.batch_size)


mb_size = args.batch_size
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 50000
log_interval = 1000
z_dim = h_dim
c_dim = args.num_classes

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(),
    cnn_filters=args.filters, cnn_units=args.units,
    freeze_embeddings=args.freeze_emb, gpu=args.gpu
)

if args.use_saved:
    model.load_state_dict(torch.load('models/vae' + utils.getModelName(args)
                                     + '.bin'))


def main():
    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

    trainer = optim.Adam(model.vae_params, lr=lr)

    for it in range(n_iter):
        inputs, labels = dataset.next_batch(args.gpu)

        recon_loss, kl_loss = model.forward(inputs)
        loss = recon_loss + kld_weight * kl_loss

        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = dataset.idxs2sentence(sample_idxs)

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; \
                   Grad_norm: {:.4f};'
                  .format(it, loss.data[0], recon_loss.data[0],
                          kl_loss.data[0], grad_norm))

            print('Sample: "{}"'.format(sample_sent))
            print()

        # Anneal learning rate
        new_lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    if args.use_saved:
        torch.save(model.state_dict(), ('models/vae' + utils.getModelName(args)
                                        + '.bin'))
    else:
        torch.save(model.state_dict(), ('models/vae'
                                        + utils.getModelName(args, 'd2')
                                        + '.bin'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()
