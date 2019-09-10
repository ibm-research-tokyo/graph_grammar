#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Apr 19 2018"


import abc
import numpy as np
import torch
import torch.nn.functional as F
from .graph import (MolecularProdRuleEmbedding,
                    MolecularProdRuleEmbeddingLastLayer,
                    MolecularProdRuleEmbeddingUsingFeatures)
from .loss import VAELoss, GrammarVAELoss
from .encoder import GRUEncoder, LSTMEncoder, FullConnectedEncoder
from .decoder import GRUDecoder, LSTMDecoder
from ..graph_grammar.symbols import NTSymbol
from copy import deepcopy
from torch import nn
from torch.autograd import Variable
from torch.optim import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau


activations = {'relu': F.relu, 'softmax': F.softmax}

loss_catalog = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'GrammarVAELoss': GrammarVAELoss,
    'VAELoss': VAELoss}

encoder_catalog = {
    'GRU': GRUEncoder,
    'LSTM': LSTMEncoder,
    'FullConnected': FullConnectedEncoder
}

decoder_catalog = {
    'GRU': GRUDecoder,
    'LSTM': LSTMDecoder
}


class Seq2SeqBase(nn.Module):

    def __init__(self, vocab_size, batch_size,
                 criterion_func='CrossEntropyLoss',
                 criterion_func_kwargs={},
                 padding_idx=None, use_gpu=False,
                 no_dropout=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu
        self.no_dropout = no_dropout
        if self.padding_idx is None:
            self.criterion = loss_catalog[criterion_func](**criterion_func_kwargs)
        else:
            self.criterion = loss_catalog[criterion_func](
                ignore_index=int(np.mod(self.padding_idx, vocab_size)),
                **criterion_func_kwargs)

    @abc.abstractmethod
    def init_hidden(self):
        pass

    @abc.abstractmethod
    def forward(self, in_seq, out_seq):
        ''' forward model

        Parameters
        ----------
        in_seq : Variable, shape (batch_size, length)
            each element corresponds to word index.
            where the index should be less than `vocab_size`

        Returns
        -------
        Variable, shape (batch_size, length, vocab_size)
            logit of each word (applying softmax yields the probability)
        '''
        pass

    def loss_func(self, pred_batch, batch_var):
        batch_var.contiguous()
        pred_batch.contiguous()
        return self.criterion(pred_batch.view(-1, self.vocab_size),
                              batch_var.view(-1))

    def batch_loss(self, batch):
        if type(batch) == list:
            in_batch, out_batch = batch
            in_batch = torch.LongTensor(np.mod(in_batch, self.vocab_size))
            out_batch = torch.LongTensor(np.mod(out_batch, self.vocab_size))
            self.init_hidden()
            in_batch_var = Variable(in_batch, requires_grad=False)
            out_batch_var = Variable(out_batch, requires_grad=False)
            if self.use_gpu:
                in_batch_var = in_batch_var.cuda()
                out_batch_var = out_batch_var.cuda()
            pred_batch = self.forward(in_batch_var, out_batch_var) # pred_batch may contain several objects
            loss = self.loss_func(pred_batch, out_batch_var)
        else:
            batch = torch.LongTensor(np.mod(batch, self.vocab_size))
            self.init_hidden()
            batch_var = Variable(batch, requires_grad=False)
            if self.use_gpu:
                batch_var = batch_var.cuda()
            pred_batch = self.forward(batch_var)
            loss = self.loss_func(pred_batch, batch_var)
        return loss

    def fit(self, data_loader_train, data_loader_val=None, max_num_examples=None,
            print_freq=1000, num_epochs=10, sgd=Adagrad, sgd_kwargs={}, logger=print):
        ''' fit to the data

        Parameters
        ----------
        data_loader_train : DataLoader
            if enumerated, it returns array-like object of shape (batch_size, length),
            where each element corresponds to word index.
        data_loader_val : DataLoader
            data loader for validation.
        print_freq : int
            how frequent to print loss
        num_epochs : int
            the number of epochs
        '''

        optimizer = sgd(self.parameters(), **sgd_kwargs)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)
        i = 0
        running_loss = 0
        finish = False
        for epoch in range(num_epochs):
            if not finish:
                for each_idx, each_batch in enumerate(data_loader_train):
                    if len(each_batch[0]) < self.batch_size:
                        each_batch[0] = torch.cat([each_batch[0],
                                                   self.padding_idx * torch.ones((self.batch_size - len(each_batch[0]),
                                                                                  len(each_batch[0][0])), dtype=torch.int64)], dim=0)
                        each_batch[1] = torch.cat([each_batch[1],
                                                   self.padding_idx * torch.ones((self.batch_size - len(each_batch[1]),
                                                                                  len(each_batch[1][0])), dtype=torch.int64)], dim=0)
                    optimizer.zero_grad()
                    loss = self.batch_loss(each_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    i += 1
                    if i % print_freq == print_freq-1:
                        msg = f'epoch: {epoch + 1}\t '\
                              f'total examples: {(i + 1) * self.batch_size}\t '\
                              f'per-example loss: {running_loss / (print_freq * self.batch_size)}'
                        logger(msg)
                        scheduler.step(running_loss)
                        running_loss = 0.0
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    if max_num_examples is not None:
                        if (i + 1) * self.batch_size > max_num_examples:
                            finish = True
                            break


                if data_loader_val is not None:
                    with torch.no_grad():
                        running_val_loss = 0.0
                        for each_idx, each_batch in enumerate(data_loader_val):
                            if len(each_batch[0]) < self.batch_size:
                                each_batch[0] = torch.cat([each_batch[0],
                                                           self.padding_idx * torch.ones((self.batch_size - len(each_batch[0]),
                                                                                          len(each_batch[0][0])), dtype=torch.int64)], dim=0)
                                each_batch[1] = torch.cat([each_batch[1],
                                                           self.padding_idx * torch.ones((self.batch_size - len(each_batch[1]),
                                                                                          len(each_batch[1][0])), dtype=torch.int64)], dim=0)

                            val_loss = self.batch_loss(each_batch)
                            running_val_loss += val_loss.item()
                    msg = f'epoch: {epoch + 1}\t '\
                          f'per-example val_loss: {running_val_loss / (self.batch_size * each_idx)}'
                    logger(msg)
                    #scheduler.step(running_val_loss)

        logger(' ** Finished Training **')
        return running_loss, running_val_loss


class Seq2SeqAutoencoder(Seq2SeqBase):

    ''' sequence-to-sequence autoencoder.
    ***** NOT VARIATIONAL ONE *****
    '''

    def __init__(self, vocab_size,
                 src_word_embedding_dim, tgt_word_embedding_dim,
                 enc_hidden_dim, dec_hidden_dim, batch_size,
                 criterion_func='CrossEntropyLoss',
                 criterion_func_kwargs={},
                 padding_idx=None, enc_num_layers=1, dec_num_layers=1,
                 dropout=0., use_gpu=False):
        super().__init__(vocab_size=vocab_size,
                         batch_size=batch_size,
                         padding_idx=padding_idx,
                         use_gpu=use_gpu,
                         criterion_func=criterion_func,
                         criterion_func_kwargs=criterion_func_kwargs)
        self.src_word_embedding_dim = src_word_embedding_dim
        self.tgt_word_embedding_dim = tgt_word_embedding_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.dropout = dropout

        self.src_embedding = nn.Embedding(
            vocab_size,
            src_word_embedding_dim,
            padding_idx=int(np.mod(self.padding_idx, vocab_size))
        )
        self.tgt_embedding = nn.Embedding(
            vocab_size,
            tgt_word_embedding_dim,
            padding_idx=int(np.mod(self.padding_idx, vocab_size))
        )

        self.encoder = nn.LSTM(input_size=self.src_word_embedding_dim,
                               hidden_size=self.enc_hidden_dim,
                               num_layers=self.enc_num_layers,
                               batch_first=True,
                               dropout=self.dropout)

        self.decoder = nn.LSTM(input_size=self.tgt_word_embedding_dim,
                               hidden_size=self.dec_hidden_dim,
                               num_layers=self.dec_num_layers,
                               batch_first=True,
                               dropout=self.dropout)

        self.enc2dec = nn.Linear(self.enc_hidden_dim,
                                 self.dec_hidden_dim)

        self.dec2vocab = nn.Linear(self.dec_hidden_dim,
                                   self.vocab_size)
        self.init_weights()
        self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            self.h0_enc = Variable(torch.zeros(self.enc_num_layers,
                                               self.batch_size,
                                               self.enc_hidden_dim).cuda(),
                                   requires_grad=False)
            self.c0_enc = Variable(torch.zeros(self.enc_num_layers,
                                               self.batch_size,
                                               self.enc_hidden_dim).cuda(),
                                   requires_grad=False)
            self.h0_dec = Variable(torch.zeros(self.dec_num_layers,
                                               self.batch_size,
                                               self.dec_hidden_dim).cuda(),
                                   requires_grad=False)
            self.c0_dec = Variable(torch.zeros(self.dec_num_layers,
                                               self.batch_size,
                                               self.dec_hidden_dim).cuda(),
                                   requires_grad=False)
        else:
            self.h0_enc = Variable(torch.zeros(self.enc_num_layers,
                                               self.batch_size,
                                               self.enc_hidden_dim),
                                   requires_grad=False)
            self.c0_enc = Variable(torch.zeros(self.enc_num_layers,
                                               self.batch_size,
                                               self.enc_hidden_dim),
                                   requires_grad=False)
            self.h0_dec = Variable(torch.zeros(self.dec_num_layers,
                                               self.batch_size,
                                               self.dec_hidden_dim),
                                   requires_grad=False)
            self.c0_dec = Variable(torch.zeros(self.dec_num_layers,
                                               self.batch_size,
                                               self.dec_hidden_dim),
                                   requires_grad=False)

    def init_weights(self, init_range=1.0):
        ''' initialize weights
        '''
        if hasattr(self.src_embedding, 'weight'):
            self.src_embedding.weight.data.uniform_(-init_range, init_range)
        if hasattr(self.tgt_embedding, 'weight'):
            self.tgt_embedding.weight.data.uniform_(-init_range, init_range)
        self.enc2dec.bias.data.fill_(0)
        self.dec2vocab.bias.data.fill_(0)

    def forward(self, in_seq, out_seq=None):
        ''' forward model

        Parameters
        ----------
        in_seq : Variable, shape (batch_size, length)
            each element corresponds to word index.
            where the index should be less than `vocab_size`

        Returns
        -------
        Variable, shape (batch_size, length, vocab_size)
            logit of each word (applying softmax yields the probability)
        '''
        src_emb = self.src_embedding(in_seq)
        tgt_emb = self.tgt_embedding(in_seq)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_enc, self.c0_enc))

        # take the last layer
        h_t = src_h_t[-1]
        c_t = src_c_t[-1]

        # fill in the first layer
        self.h0_dec[0] = nn.Tanh()(self.enc2dec(h_t))
        self.c0_dec[0] = nn.Tanh()(self.enc2dec(c_t))

        tgt_h, (_, _) = self.decoder(tgt_emb, (self.h0_dec, self.c0_dec))
        vocab_logit = self.dec2vocab(tgt_h)
        return vocab_logit

    def logit2prob(self, logits):
        ''' compute probability distribution from logits.

        Parameters
        ----------
        logits : Variable, shape (batch_size, length, vocab_size)

        Returns
        -------
        probs : Variable, shape (batch_size, length, vocab_size)
        '''
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqVAE(Seq2SeqBase):

    def __init__(self, vocab_size, latent_dim=64, max_len=80,
                 batch_size=64, padding_idx=-1, use_gpu=False,
                 start_rule_embedding=True,
                 encoder_name='GRU',
                 encoder_params={'hidden_dim': 384, 'num_layers': 3, 'bidirectional': True,
                                 'dropout': 0.1},
                 decoder_name='GRU',
                 decoder_params={'hidden_dim': 384, 'num_layers': 2,
                                 'dropout': 0.1},
                 prod_rule_embed_params={'out_dim': 128},
                 criterion_func='VAELoss',
                 criterion_func_kwargs={'beta': 1.0},
                 no_dropout=False):
        super().__init__(vocab_size=vocab_size,
                         batch_size=batch_size,
                         padding_idx=padding_idx,
                         use_gpu=use_gpu,
                         criterion_func=criterion_func,
                         criterion_func_kwargs=criterion_func_kwargs,
                         no_dropout=no_dropout)
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu
        self.encoder_name = encoder_name
        self.encoder_params = encoder_params
        self.decoder_name = decoder_name
        self.decoder_params = decoder_params
        self.prod_rule_embed_params = prod_rule_embed_params
        self.start_rule_embedding = start_rule_embedding
        
        self.init_embed()
        self.init_vae()
        self.init_weights()
        self.init_hidden()

    def init_embed(self):
        self.src_embedding = nn.Embedding(
            self.vocab_size,
            self.prod_rule_embed_params['out_dim'],
            padding_idx=None if self.padding_idx is None else int(np.mod(self.padding_idx, self.vocab_size))
        )
        self.tgt_embedding = nn.Embedding(
            self.vocab_size,
            self.prod_rule_embed_params['out_dim'],
            padding_idx=None if self.padding_idx is None else int(np.mod(self.padding_idx, self.vocab_size))
        )
        pass

    def init_vae(self):
        self.encoder = encoder_catalog[self.encoder_name](
            input_dim=self.prod_rule_embed_params['out_dim'],
            batch_size=self.batch_size,
            use_gpu=self.use_gpu,
            no_dropout=self.no_dropout,
            **self.encoder_params)

        if self.start_rule_embedding:
            self.emb2mean = nn.Linear(self.prod_rule_embed_params['out_dim'], self.latent_dim//2, bias=False)
            self.emb2logvar = nn.Linear(self.prod_rule_embed_params['out_dim'], self.latent_dim//2)
            self.hidden2mean = nn.Linear((self.encoder_params.get('bidirectional', False) + 1) * self.encoder_params['hidden_dim'],
                                         self.latent_dim//2,
                                         bias=False)
            self.hidden2logvar = nn.Linear((self.encoder_params.get('bidirectional', False) + 1) * self.encoder_params['hidden_dim'],
                                           self.latent_dim//2)
        else:
            self.hidden2mean = nn.Linear((self.encoder_params.get('bidirectional', False) + 1) * self.encoder_params['hidden_dim'],
                                         self.latent_dim,
                                         bias=False)
            self.hidden2logvar = nn.Linear((self.encoder_params.get('bidirectional', False) + 1) * self.encoder_params['hidden_dim'],
                                           self.latent_dim)

        self.decoder = decoder_catalog[self.decoder_name](
            input_dim=self.prod_rule_embed_params['out_dim'],
            batch_size=self.batch_size,
            use_gpu=self.use_gpu,
            no_dropout=self.no_dropout,
            **self.decoder_params)
        self.latent2tgt_emb = nn.Linear(self.latent_dim,
                                        self.prod_rule_embed_params['out_dim'])
        self.latent2hidden_dict = {}
        for each_hidden in self.decoder.hidden_dict.keys():
            self.latent2hidden_dict[each_hidden] = nn.Linear(self.latent_dim, self.decoder_params['hidden_dim'])
            if self.use_gpu:
                self.latent2hidden_dict[each_hidden] = self.latent2hidden_dict[each_hidden].cuda()


        self.dec2vocab = nn.Linear(self.decoder_params['hidden_dim'],
                                   self.vocab_size)
        pass

    def init_hidden(self):
        self.encoder.init_hidden()
        self.decoder.init_hidden()

    def init_weights(self, init_range=0.1):
        ''' initialize weights
        '''
        if hasattr(self.src_embedding, 'weight'):
            self.src_embedding.weight.data.uniform_(-init_range, init_range)
        if hasattr(self.tgt_embedding, 'weight'):
            self.tgt_embedding.weight.data.uniform_(-init_range, init_range)

    def encode(self, in_seq):
        src_emb = self.src_embedding(in_seq)
        src_h = self.encoder.forward(src_emb)
        if self.encoder_params.get('bidirectional', False):
            concat_src_h = torch.cat((src_h[:, -1, 0, :], src_h[:, 0, 1, :]), dim=1)
            if self.start_rule_embedding:
                return torch.cat((self.hidden2mean(concat_src_h), self.emb2mean(src_emb[:, -1, :])), dim=1),\
                    torch.cat((self.hidden2logvar(concat_src_h), self.emb2logvar(src_emb[:, -1, :])), dim=1)
            else:
                return self.hidden2mean(concat_src_h),\
                    self.hidden2logvar(concat_src_h)
        else:
            if self.start_rule_embedding:
                return torch.cat((self.hidden2mean(src_h[:, -1, :]), self.emb2mean(src_emb[:, -1, :])), dim=1),\
                    torch.cat((self.hidden2logvar(src_h[:, -1, :]), self.emb2logvar(src_emb[:, -1, :])), dim=1)
            else:
                return self.hidden2mean(src_h[:, -1, :]),\
                    self.hidden2logvar(src_h[:, -1, :])

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            if self.use_gpu:
                eps = Variable(std.data.new(std.size()).normal_()).cuda()
            else:
                eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, out_seq=None, deterministic=False):
        hidden_dict_0 = {}
        for each_hidden in self.latent2hidden_dict.keys():
            hidden_dict_0[each_hidden] = self.latent2hidden_dict[each_hidden](z)
        self.decoder.feed_hidden(hidden_dict_0)

        if out_seq is not None:
            tgt_emb0 = self.latent2tgt_emb(z)
            tgt_emb0 = tgt_emb0.view(tgt_emb0.shape[0], 1, tgt_emb0.shape[1])
            tgt_emb = torch.cat((tgt_emb0, self.tgt_embedding(out_seq)), dim=1)[:, :-1, :]
            tgt_emb_pred_list = []
            for each_idx in range(self.max_len):
                tgt_emb_pred = self.decoder.forward_one_step(tgt_emb[:, each_idx, :].view(self.batch_size, 1, -1))
                tgt_emb_pred_list.append(tgt_emb_pred)
            vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
            return vocab_logit
        else:
            with torch.no_grad():
                tgt_emb = self.latent2tgt_emb(z)
                tgt_emb = tgt_emb.view(tgt_emb.shape[0], 1, tgt_emb.shape[1])
                tgt_emb_pred_list = []
                for _ in range(self.max_len):
                    tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
                    tgt_emb_pred_list.append(tgt_emb_pred)
                    vocab_logit = self.dec2vocab(tgt_emb_pred)
                    for each_batch_idx in range(self.batch_size):
                        #if deterministic:
                        tgt_id = np.argmax(vocab_logit[each_batch_idx, 0, :])
                        #else:
                        #    tgt_id = np.random.choice(list(np.range(vocab_logit.shape[2])),
                        #                              1,
                        #                              F.softmax(vocab_logit[each_batch_idx, 0, :]))[0]
                        indice_tensor = torch.LongTensor([tgt_id])
                        if self.use_gpu:
                            indice_tensor = indice_tensor.cuda()
                        tgt_emb[each_batch_idx, :] = self.tgt_embedding(indice_tensor)
                vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
                return vocab_logit

    def forward(self, in_seq, out_seq):
        ''' forward model

        Parameters
        ----------
        in_seq : Variable, shape (batch_size, length)
            each element corresponds to word index.
            where the index should be less than `vocab_size`

        Returns
        -------
        Variable, shape (batch_size, length, vocab_size)
            logit of each word (applying softmax yields the probability)
        '''
        mu, logvar = self.encode(in_seq)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, out_seq), mu, logvar

    def logit2prob(self, logits):
        ''' compute probability distribution from logits.

        Parameters
        ----------
        logits : Variable, shape (batch_size, length, vocab_size)

        Returns
        -------
        probs : Variable, shape (batch_size, length, vocab_size)
        '''
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs

    def loss_func(self, pred, tgt):
        return self.criterion(pred[0], tgt,
                              pred[1], pred[2])


class GrammarSeq2SeqVAE(Seq2SeqVAE):

    '''
    Variational seq2seq with grammar.
    TODO: rewrite this class using mixin
    '''

    def __init__(self, hrg, latent_dim=64, max_len=80,
                 batch_size=64, padding_idx=-1, use_gpu=False,
                 start_rule_embedding=True,
                 encoder_name='GRU',
                 encoder_params={'hidden_dim': 384, 'num_layers': 3, 'bidirectional': True,
                                 'dropout': 0.1},
                 decoder_name='GRU',
                 decoder_params={'hidden_dim': 384, 'num_layers': 2,
                                 'dropout': 0.1},
                 prod_rule_embed='MolecularProdRuleEmbedding',
                 prod_rule_embed_params={'out_dim': 128},
                 criterion_func='VAELoss',
                 criterion_func_kwargs={'beta': 1.0},
                 class_weight=None,
                 no_dropout=False):
        self.hrg = hrg
        self.prod_rule_corpus = hrg.prod_rule_corpus
        self.prod_rule_embed = prod_rule_embed
        self.prod_rule_embed_params = prod_rule_embed_params
        if criterion_func == 'GrammarVAELoss':
            criterion_func_kwargs = deepcopy(dict(criterion_func_kwargs))
            criterion_func_kwargs['hrg'] = hrg
            if class_weight is not None:
                criterion_func_kwargs['class_weight'] = class_weight

        vocab_size = hrg.num_prod_rule + 1
        super().__init__(vocab_size=vocab_size,
                         latent_dim=latent_dim, max_len=max_len,
                         batch_size=batch_size, padding_idx=padding_idx, use_gpu=use_gpu,
                         start_rule_embedding=start_rule_embedding,
                         encoder_name=encoder_name,
                         encoder_params=encoder_params,
                         decoder_name=decoder_name,
                         decoder_params=decoder_params,
                         prod_rule_embed_params=prod_rule_embed_params,
                         criterion_func=criterion_func,
                         criterion_func_kwargs=criterion_func_kwargs,
                         no_dropout=no_dropout)

    def sample(self, sample_size=-1, deterministic=True, return_z=False):
        self.init_hidden()
        if sample_size == -1:
            sample_size = self.batch_size

        num_iter = int(np.ceil(sample_size / self.batch_size))
        hg_list = []
        z_list = []
        for _ in range(num_iter):
            if self.use_gpu:
                z = Variable(torch.normal(
                    torch.zeros(self.batch_size, self.latent_dim),
                    torch.ones(self.batch_size * self.latent_dim))).cuda()
            else:
                z = Variable(torch.normal(
                    torch.zeros(self.batch_size, self.latent_dim),
                    torch.ones(self.batch_size * self.latent_dim)))
            _, each_hg_list = self.decode(z, deterministic=deterministic, return_hg_list=True)
            z_list.append(z)
            hg_list += each_hg_list
        z = torch.cat(z_list)[:sample_size]
        hg_list = hg_list[:sample_size]
        if return_z:
            return hg_list, z.cpu().detach().numpy()
        else:
            return hg_list

    def decode(self, z=None, out_seq=None, deterministic=True, return_hg_list=False):
        if z is None:
            if self.use_gpu:
                z = Variable(torch.normal(
                    torch.zeros(self.batch_size, self.latent_dim),
                    torch.ones(self.batch_size * self.latent_dim))).cuda()
            else:
                z = Variable(torch.normal(
                    torch.zeros(self.batch_size, self.latent_dim),
                    torch.ones(self.batch_size * self.latent_dim)))
        else:
            z = z.cuda()

        hidden_dict_0 = {}
        for each_hidden in self.latent2hidden_dict.keys():
            hidden_dict_0[each_hidden] = self.latent2hidden_dict[each_hidden](z)
        self.decoder.feed_hidden(hidden_dict_0)

        if out_seq is not None:
            tgt_emb0 = self.latent2tgt_emb(z)
            tgt_emb0 = tgt_emb0.view(tgt_emb0.shape[0], 1, tgt_emb0.shape[1])
            tgt_emb = torch.cat((tgt_emb0, self.tgt_embedding(out_seq)), dim=1)[:, :-1, :]
            tgt_emb_pred_list = []
            for each_idx in range(self.max_len):
                tgt_emb_pred = self.decoder.forward_one_step(tgt_emb[:, each_idx, :].view(self.batch_size, 1, -1))
                tgt_emb_pred_list.append(tgt_emb_pred)
            vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
            return vocab_logit
        else:
            with torch.no_grad():
                tgt_emb = self.latent2tgt_emb(z)
                tgt_emb = tgt_emb.view(tgt_emb.shape[0], 1, tgt_emb.shape[1])
                tgt_emb_pred_list = []
                stack_list = []
                hg_list = []
                nt_symbol_list = []
                nt_edge_list = []
                gen_finish_list = []
                for _ in range(self.batch_size):
                    stack_list.append([])
                    hg_list.append(None)
                    nt_symbol_list.append(NTSymbol(degree=0,
                                                   is_aromatic=False,
                                                   bond_symbol_list=[]))
                    nt_edge_list.append(None)
                    gen_finish_list.append(False)

                for idx in range(self.max_len):
                    tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
                    tgt_emb_pred_list.append(tgt_emb_pred)
                    vocab_logit = self.dec2vocab(tgt_emb_pred)
                    for each_batch_idx in range(self.batch_size):
                        if not gen_finish_list[each_batch_idx]:
                            prod_rule = self.hrg.prod_rule_corpus.sample(vocab_logit[each_batch_idx, :, :-1].squeeze().cpu().numpy(),
                                                                         nt_symbol_list[each_batch_idx],
                                                                         deterministic=deterministic)
                            tgt_id = self.hrg.prod_rule_list.index(prod_rule)
                            hg_list[each_batch_idx], nt_edges = prod_rule.applied_to(hg_list[each_batch_idx], nt_edge_list[each_batch_idx])
                            stack_list[each_batch_idx].extend(nt_edges[::-1])
                            if len(stack_list[each_batch_idx]) == 0:
                                gen_finish_list[each_batch_idx] = True
                            else:
                                nt_edge_list[each_batch_idx] = stack_list[each_batch_idx].pop()
                                nt_symbol_list[each_batch_idx] = hg_list[each_batch_idx].edge_attr(nt_edge_list[each_batch_idx])['symbol']
                        else:
                            tgt_id = np.mod(self.padding_idx, self.vocab_size)
                        indice_tensor = torch.LongTensor([tgt_id])
                        if self.use_gpu:
                            indice_tensor = indice_tensor.cuda()
                        tgt_emb[each_batch_idx, :] = self.tgt_embedding(indice_tensor)
                vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
                return vocab_logit, hg_list

    def init_embed(self):
        if self.prod_rule_embed == 'Embedding':
            self.src_embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.prod_rule_embed_params['out_dim'],
                padding_idx=None if self.padding_idx is None else int(np.mod(self.padding_idx, self.vocab_size)),
            )
            self.tgt_embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.prod_rule_embed_params['out_dim'],
                padding_idx=None if self.padding_idx is None else int(np.mod(self.padding_idx, self.vocab_size))
            )
        elif self.prod_rule_embed == 'MolecularProdRuleEmbedding':
            self.src_embedding = MolecularProdRuleEmbedding(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                element_embed_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
            self.tgt_embedding = MolecularProdRuleEmbedding(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                element_embed_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
        elif self.prod_rule_embed == 'MolecularProdRuleEmbeddingLastLayer':
            self.src_embedding = MolecularProdRuleEmbeddingLastLayer(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                element_embed_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
            self.tgt_embedding = MolecularProdRuleEmbeddingLastLayer(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                element_embed_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
        elif self.prod_rule_embed == 'MolecularProdRuleEmbeddingUsingFeatures':
            self.src_embedding = MolecularProdRuleEmbeddingUsingFeatures(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
            self.tgt_embedding = MolecularProdRuleEmbeddingUsingFeatures(
                prod_rule_corpus=self.prod_rule_corpus,
                layer2layer_activation=activations[self.prod_rule_embed_params['layer2layer_activation']],
                layer2out_activation=activations[self.prod_rule_embed_params['layer2out_activation']],
                out_dim=self.prod_rule_embed_params['out_dim'],
                num_layers=self.prod_rule_embed_params['num_layers'],
                padding_idx=self.padding_idx,
                use_gpu=self.use_gpu)
        else:
            raise NotImplementedError(f'{self.prod_rule_embed} is not implemented.')

