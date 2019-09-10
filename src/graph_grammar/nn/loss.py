#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "July 31 2018"

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class VAELoss(_Loss):

    '''
    a loss function for VAE
    '''

    def __init__(self, ignore_index=None, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(self, in_seq_pred, in_seq, mu, logvar):
        ''' compute VAE loss

        Parameters
        ----------
        in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
            logit
        in_seq : torch.Tensor, shape (batch_size, max_len)
            each element corresponds to a word id in vocabulary.
        mu : torch.Tensor, shape (batch_size, hidden_dim)
        logvar : torch.Tensor, shape (batch_size, hidden_dim)
            mean and log variance of the normal distribution
        '''
        cross_entropy = F.cross_entropy(
            in_seq_pred.view(-1, in_seq_pred.shape[2]),
            in_seq.view(-1),
            reduction='sum',
            ignore_index=self.ignore_index if self.ignore_index is not None else -100)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return cross_entropy + self.beta * kl_div


class GrammarVAELoss(_Loss):

    '''
    a loss function for Grammar VAE

    Attributes
    ----------
    hrg : HyperedgeReplacementGrammar
    ignore_index : int
        index to be ignored
    beta : float
        coefficient of KL divergence
    '''

    def __init__(self, hrg, ignore_index=None, beta=1.0, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.hrg = hrg
        self.ignore_index = ignore_index
        self.beta = beta
        self.class_weight = class_weight

    def forward(self, in_seq_pred, in_seq, mu, logvar):
        ''' compute VAE loss

        Parameters
        ----------
        in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
            logit
        in_seq : torch.Tensor, shape (batch_size, max_len)
            each element corresponds to a word id in vocabulary.
        mu : torch.Tensor, shape (batch_size, hidden_dim)
        logvar : torch.Tensor, shape (batch_size, hidden_dim)
            mean and log variance of the normal distribution
        '''
        batch_size = in_seq_pred.shape[0]
        max_len = in_seq_pred.shape[1]
        vocab_size = in_seq_pred.shape[2]
        mask = torch.zeros(in_seq_pred.shape)

        for each_batch in range(batch_size):
            for each_idx in range(max_len):
                prod_rule_idx = in_seq[each_batch, each_idx]                    
                if prod_rule_idx == self.ignore_index:
                    continue
                lhs = self.hrg.prod_rule_corpus.prod_rule_list[prod_rule_idx].lhs_nt_symbol
                lhs_idx = self.hrg.prod_rule_corpus.nt_symbol_list.index(lhs)
                mask[each_batch, each_idx, :-1] = torch.FloatTensor(self.hrg.prod_rule_corpus.lhs_in_prod_rule[lhs_idx])
        mask = mask.cuda()
        in_seq_pred = mask * in_seq_pred

        cross_entropy = F.cross_entropy(
            in_seq_pred.view(-1, vocab_size),
            in_seq.view(-1),
            weight=self.class_weight,
            reduction='sum',
            ignore_index=self.ignore_index if self.ignore_index is not None else -100)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return cross_entropy + self.beta * kl_div
