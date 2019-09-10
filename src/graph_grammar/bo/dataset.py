#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "July 18 2018"

import numpy as np
from torch.utils.data import DataLoader
from ..descriptors import log_p, synthetic_accessibility, cycle_score
from ..nn.dataset import HRGDataset

def target_func(mol):
    return 

class LatentMolSet(object):
    '''
    '''
    def __init__(self, model, hrg,
                 target_func_list=[log_p, synthetic_accessibility, cycle_score],
                 target_weight_list=[1, -1, 1],
                 target_normalize_list=[True, True, True]):
        self.model = model
        self.hrg = hrg
        self.target_func_list = target_func_list
        self.target_weight_list = target_weight_list
        self.target_normalize_list = target_normalize_list

    def construct(self, mol_list, prod_rule_seq_list):
        ''' construct a dataset

        Parameters
        ----------
        mol_list : list of Mol
        prod_rule_seq_list : list of production rule sequences
            
        '''
        self.latent_array = np.zeros((len(mol_list), self.model.latent_dim))
        hrg_dataset = HRGDataset(self.hrg, prod_rule_seq_list, self.model.max_len)
        hrg_dataloader = DataLoader(hrg_dataset, batch_size=self.model.batch_size,
                                    shuffle=False, drop_last=False)

        for each_idx, each_batch in enumerate(hrg_dataloader):
            mean_array, _ = self.model.encode(each_batch)
            self.latent_array[self.model.batch_size * each_idx
                              : min(self.model.batch_size * (each_idx + 1), len(mol_list)), :]\
                              = mean_array
        # calculate target values
        tgt_array_list = []
        normalized_tgt_array_list = []
        for each_idx, each_func in enumerate(self.target_func_list):
            tgt_array = []
            for each_mol in mol_list:
                tgt_array.append(each_func(each_mol))
            tgt_array_list.append(np.array(tgt_array))
            if self.target_normalize_list[each_idx]:
                normalized_tgt_array_list.append(
                    (np.array(tgt_array) - np.mean(np.array(tgt_array))) \
                    / np.std(np.array(tgt_array)))
            else:
                normalized_tgt_array_list.append((np.array(tgt_array)))
        self.target_array = self.target_weight_list @ normalized_tgt_array_list
        return self.latent_array, self.target_array
