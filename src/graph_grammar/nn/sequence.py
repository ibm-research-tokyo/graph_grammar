#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Apr 9 2018"

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adagrad
from typing import List


class LSTM(nn.Module):
    ''' LSTM model

    Attributes
    ----------
    input_size : int
        dimension of the input sequence
    hidden_size : int
        dimension of the hidden sequence
    batch_size : int
        batch size
    num_layers : int
        the number of LSTM layers
    dropout : float
        if nonzero, a dropout layer will be introduced after the LSTM output
    use_gpu : bool
        if True, the hidden states are CUDA-compatible.
    '''
    def __init__(self, input_size, hidden_size, batch_size,
                 num_layers=1, dropout=.1, use_gpu=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        # init lstm
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)

        # init hidden states
        self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            self.h0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.hidden_size).cuda(),
                               requires_grad=False)
            self.c0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.hidden_size).cuda(),
                               requires_grad=False)
        else:
            self.h0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.hidden_size),
                               requires_grad=False)
            self.c0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.hidden_size),
                               requires_grad=False)

    def forward(self, x):
        ''' forward model

        Parameters
        ----------
        x : Variable, shape (length, batch_size, input_size)
        '''
        h, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        return h

    def fit(self, seq_list: List, objective='cross_entropy',
            print_freq=1000, num_epochs=10, sgd_kwargs={}):
        ''' train LSTM using DataLoader

        Parameters
        ----------
        seq_list : list
            each element corresponds to a sequence
        objective : str
            objective function
        print_freq : int
            how frequently loss is printed
        num_epochs : int
            the number of training epochs
        sgd_kwargs : dict
            keywords fed into SGD
        '''
        if objective == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif objective == 'mse':
            criterion = nn.MSELoss()
        elif objective == 'nll': # nll stands for negative log-likelihood
            criterion = nn.NLLLoss()
        else:
            raise NotImplementedError

        optimizer = Adagrad(self.parameters(), **sgd_kwargs)
        i = 0
        running_loss = 0
        for epoch in range(num_epochs):
            for each_idx in range(0, len(seq_list), self.batch_size):
                each_seq = torch.stack(
                    seq_list[each_idx:each_idx + self.batch_size], dim=1)
                seq = Variable(each_seq, requires_grad=False)

                optimizer.zero_grad()
                pred_seq = self.forward(seq[:-1])
                loss = criterion(pred_seq, seq[:-1])
                loss.backward()
                optimizer.step()
                self.init_hidden()

                # print statistics
                running_loss += loss.data[0]
                i += 1
                if i % print_freq == print_freq-1:
                    print('epoch: {}\t total examples: {}\t loss: {}'.format(
                        epoch + 1, i + 1, running_loss / print_freq))
                    running_loss = 0.0

        print('Finished Training')


class WordLSTM(nn.Module):

    ''' language model using LSTM

    Attributes
    ----------
    vocab_size : int
        size of vocabulary, including pad symbol if exists.
    word_embed_size : int
        dimension of word embedding
    batch_size : int
        mini-batch size
    num_layers : int
        the number of layers in LSTM.
    padding_idx : int
        index used for padding. if None, no padding.
    dropout : float
        dropout ratio
    use_gpu : bool
        if True, use GPU.
    '''

    def __init__(self, vocab_size, word_embed_size, batch_size, num_layers=1,
                 padding_idx=None, dropout=.1, use_gpu=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.in_embed = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=word_embed_size,
                                     padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(input_size=word_embed_size,
                            hidden_size=word_embed_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.out_linear = nn.Linear(in_features=word_embed_size,
                                    out_features=vocab_size)
        # init hidden states
        self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            self.h0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.word_embed_size).cuda(),
                               requires_grad=False)
            self.c0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.word_embed_size).cuda(),
                               requires_grad=False)
        else:
            self.h0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.word_embed_size),
                               requires_grad=False)
            self.c0 = Variable(torch.zeros(self.num_layers,
                                           self.batch_size,
                                           self.word_embed_size),
                               requires_grad=False)

    def forward(self, x):
        ''' forward model

        Parameters
        ----------
        x : Variable, shape (batch_size, length)
            each element corresponds to word index, 
            where the index should be less than `vocab_size`

        Returns
        -------
        Variable, shape (batch_size, length)
            
        '''
        in_embed = self.in_embed(x)
        out_embed, (self.h0, self.c0) = self.lstm(in_embed, (self.h0, self.c0))
        out_score = self.out_linear(out_embed)
        return out_score

    def fit(self, data_loader, print_freq=1000, num_epochs=10):
        ''' fit to the data

        Parameters
        ----------
        data_loader : DataLoader
            if enumerated, it returns array-like object of shape (batch_size, length),
            where each element corresponds to word index.
        print_freq : int
            how frequent to print loss
        num_epochs : int
            the number of epochs
        '''

        def repackage_hidden(h):
            """Wraps hidden states in new Variables, to detach them from their history."""
            if type(h) == Variable:
                return Variable(h.data)
            else:
                return tuple(repackage_hidden(v) for v in h)

        if self.padding_idx is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        optimizer = Adagrad(self.parameters())
        i = 0
        running_loss = 0
        for epoch in range(num_epochs):
            for each_idx, each_batch in enumerate(data_loader):
                batch_var = Variable(each_batch, requires_grad=False)
                if self.use_gpu:
                    batch_var = batch_var.cuda()

                try:
                    pred_batch = self.forward(batch_var[:, :-1])
                except:
                    import ipdb; ipdb.set_trace()
                    
                pred_batch.contiguous()
                batch_var.contiguous()
                tgt = batch_var[:, :-1]
                tgt.contiguous()
                loss = criterion(pred_batch.view(-1, self.vocab_size),
                                 tgt.view(-1))
                loss.backward()
                optimizer.step()
                self.init_hidden()

                # print statistics
                running_loss += loss.data[0]
                i += 1
                if i % print_freq == print_freq-1:
                    print('epoch: {}\t total examples: {}\t loss: {}'.format(
                        epoch + 1, (i + 1) * self.batch_size, running_loss / print_freq))
                    running_loss = 0.0

        print('Finished Training')


if __name__ == '__main__':
    '''
    input_size = 10
    hidden_size = 10
    length = 10
    batch_size = 10
    num_layers = 1
    dropout = 0.05
    in_seq = [torch.from_numpy((
        np.hstack([np.sin(0.1 * np.arange(length)) + 0 * np.random.randn(length)]
                  * input_size)).reshape(length, input_size).astype(np.float32))] * batch_size * 1000
    model = LSTM(input_size=input_size,
                 hidden_size=hidden_size,
                 batch_size=batch_size,
                 num_layers=num_layers,
                 dropout=dropout)
    model.init_hidden()
    model.fit(in_seq, 'mse', num_epochs=1000, sgd_kwargs={'lr' : 1.0})
    '''
    batch_size = 2
    model = WordLSTM(10, 5, batch_size)
    in_seq = [np.arange(100) % 10] * 10000
    from torch.utils.data import DataLoader
    data_loader = DataLoader(in_seq, batch_size=batch_size)
    model.fit(data_loader)
