#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 24 2018"


# set luigi_config_path BEFORE importing luigi
import argparse
import os
import sys
try:
    working_dir = sys.argv[1:][sys.argv[1:].index("--working-dir") + 1]
    os.chdir(working_dir)
except ValueError:
    raise argparse.ArgumentError("--working-dir option must be specified.")
# add a path to luigi.cfg
os.environ["LUIGI_CONFIG_PATH"] = os.path.abspath(os.path.join("INPUT", "luigi.cfg"))
sys.path.append(os.path.abspath(os.path.join("INPUT")))
from param import (DataPreprocessing_params, Train_params, CheckReconstructionRate_params,
                   ConstructDatasetForBO_params, ComputeTargetValues_params, Sample_params,
                   BayesianOptimization_params, Encode_params,
                   GuacaMol_params, 
                   MultipleBayesianOptimization_params,
                   TrainWithPred_params, ConstrainedMolOpt_params)

# imports
from collections import Counter
from copy import deepcopy
from datetime import datetime, date
from luigine.abc import MainTask, AutoNamingTask, main
from graph_grammar.algo.tree_decomposition import (tree_decomposition,
                                                   tree_decomposition_with_hrg,
                                                   tree_decomposition_from_leaf,
                                                   topological_tree_decomposition,
                                                   molecular_tree_decomposition)
from graph_grammar.bo.mol_opt import MolecularOptimization
from graph_grammar.descriptors import log_p, synthetic_accessibility, cycle_score, synthetic_accessibility_batch, load_fscores
from graph_grammar.graph_grammar.hrg import HyperedgeReplacementGrammar as HRG
from graph_grammar.io.smi import HGGen, hg_to_mol
from graph_grammar.nn.dataset import HRGDataset, batch_padding
from graph_grammar.nn.sequence import WordLSTM
from graph_grammar.nn.autoencoder import (Seq2SeqAutoencoder, Seq2SeqVAE,
                                          GrammarSeq2SeqVAE)
from graph_grammar.nn.autoencoder_with_predictor import GrammarSeq2SeqVAEWithPred
from graph_grammar.nn.loss import GrammarVAELoss, VAELoss
from graph_grammar.benchmarks.guacamol import MHGDirectedGenerator, MHGDistributionMatchingGenerator
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adagrad, RMSprop, Adam
from guacamol.goal_directed_generator import GoalDirectedGenerator
import glob
import gzip
import luigi
import logging
import numpy as np
import pandas as pd
import pickle
import torch
import traceback


logger = logging.getLogger('luigi-interface')

td_catalog = {
    'tree_decomposition': tree_decomposition,
    'tree_decomposition_with_hrg': tree_decomposition_with_hrg,
    'tree_decomposition_from_leaf': tree_decomposition_from_leaf,
    'topological_tree_decomposition': topological_tree_decomposition,
    'molecular_tree_decomposition': molecular_tree_decomposition
}

ae_catalog = {
    'GrammarSeq2SeqVAE': GrammarSeq2SeqVAE,
    'GrammarSeq2SeqVAEWithPred': GrammarSeq2SeqVAEWithPred
}

sgd_catalog = {
    'Adagrad': Adagrad,
    'Adam': Adam,
    'RMSprop': RMSprop
}


def get_dataloaders(hrg, prod_rule_seq_list, train_params,
                    target_val_list=None, batch_size=None, shuffle=True):
    ''' return a dataloader for train/val/test

    Parameters
    ----------
    prod_rule_seq_list : List of lists
        each element corresponds to a sequence of production rules.
    train_params : dict
        self.Train_params

    Returns
    -------
    Dataloaders for train, val, test of autoencoders
        each batch contains two torch Tensors, each of which corresponds to input and output of autoencoder.
    '''
    if batch_size is None:
        batch_size = train_params['model_params']['batch_size']
    prod_rule_seq_list_train = prod_rule_seq_list[: train_params['num_train']]
    prod_rule_seq_list_val = prod_rule_seq_list[train_params['num_train']
                                                : train_params['num_train'] + train_params['num_val']]
    prod_rule_seq_list_test = prod_rule_seq_list[train_params['num_train'] + train_params['num_val']
                                                 : train_params['num_train']
                                                 + train_params['num_val']
                                                 + train_params['num_test']]
    if target_val_list is None:
        target_val_list_train = None
        target_val_list_val = None
        target_val_list_test = None
    else:
        target_val_list_train = target_val_list[: train_params['num_train']]
        target_val_list_val = target_val_list[train_params['num_train']
                                              : train_params['num_train'] + train_params['num_val']]
        target_val_list_test = target_val_list[train_params['num_train'] + train_params['num_val']
                                               : train_params['num_train']
                                               + train_params['num_val']
                                               + train_params['num_test']]
    hrg_dataset_train = HRGDataset(hrg,
                                   prod_rule_seq_list_train,
                                   train_params['model_params']['max_len'],
                                   target_val_list=target_val_list_train,
                                   inversed_input=train_params['inversed_input'])
    hrg_dataloader_train = DataLoader(dataset=hrg_dataset_train,
                                      batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
    if train_params['num_val'] != 0:
        hrg_dataset_val = HRGDataset(hrg,
                                     prod_rule_seq_list_val,
                                     train_params['model_params']['max_len'],
                                     target_val_list=target_val_list_val,
                                     inversed_input=train_params['inversed_input'])
        hrg_dataloader_val = DataLoader(dataset=hrg_dataset_val,
                                        batch_size=batch_size,
                                        shuffle=shuffle, drop_last=False)
    else:
        hrg_dataset_val = None
        hrg_dataloader_val = None
    if train_params['num_test'] != 0 :
        hrg_dataset_test = HRGDataset(hrg,
                                      prod_rule_seq_list_test,
                                      train_params['model_params']['max_len'],
                                      target_val_list=target_val_list_test,
                                      inversed_input=train_params['inversed_input'])
        hrg_dataloader_test = DataLoader(dataset=hrg_dataset_test,
                                         batch_size=batch_size,
                                         shuffle=shuffle, drop_last=False)
    else:
        hrg_dataset_test = None
        hrg_dataloader_test = None
    return hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test



class DataPreprocessing(AutoNamingTask):
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    working_subdir = luigi.Parameter(default="data_prep")

    def requires(self):
        return []

    def run(self):
        hg_list = HGGen(os.path.join("INPUT", 'data', "train_val_test.txt"),
                        kekulize=self.DataPreprocessing_params['kekulize'],
                        add_Hs=self.DataPreprocessing_params['add_Hs'],
                        all_single=self.DataPreprocessing_params['all_single'])
        hrg = HRG(tree_decomposition=td_catalog[self.DataPreprocessing_params['tree_decomposition']],
                  ignore_order=self.DataPreprocessing_params['ignore_order'],
                  **self.DataPreprocessing_params['tree_decomposition_kwargs'])
        prod_rule_seq_list = hrg.learn(hg_list, logger=logger.info, max_mol=self.DataPreprocessing_params.get('max_mol', np.inf))
        logger.info(" * the number of prod rules is {}".format(hrg.num_prod_rule))

        if self.DataPreprocessing_params.get('draw_prod_rule', False):
            if not os.path.exists(os.path.join('OUTPUT', self.working_subdir, 'prod_rules')):
                os.mkdir(os.path.join('OUTPUT', self.working_subdir, 'prod_rules'))
            for each_idx, each_prod_rule in enumerate(hrg.prod_rule_corpus.prod_rule_list):
                each_prod_rule.draw(os.path.join('OUTPUT', self.working_subdir, 'prod_rules', f'{each_idx}'))
        with gzip.open(self.output().path, "wb") as f:
            pickle.dump((hrg, prod_rule_seq_list), f)

    def load_output(self):
        with gzip.open(self.output().path, "rb") as f:
            hrg, prod_rule_seq_list = pickle.load(f)
        return hrg, prod_rule_seq_list


class DataPreprocessing4ConstrainedMolOpt(AutoNamingTask):
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    working_subdir = luigi.Parameter(default='data_prep_4_const_mol_opt')

    def requires(self):
        return DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params)

    def run(self):
        hrg, prod_rule_seq_list = self.requires().load_output()

        with open(os.path.join('INPUT', 'data', 'opt.valid.logP-SA')) as f_opt:
            valid_dict = {}
            for each_line in f_opt:
                each_smiles = each_line.split()[0]
                with open(os.path.join('INPUT', 'data', 'train_val_test.txt')) as f:
                    for each_idx, each_line in enumerate(f):
                        if each_smiles == each_line.split()[0]:
                            valid_dict[each_smiles] = each_idx
                            break
        with open(os.path.join('INPUT', 'data', 'opt.test.logP-SA')) as f_opt:
            test_dict = {}
            for each_line in f_opt:
                each_smiles = each_line.split()[0]
                with open(os.path.join('INPUT', 'data', 'train_val_test.txt')) as f:
                    for each_idx, each_line in enumerate(f):
                        if each_smiles == each_line.split()[0]:
                            test_dict[each_smiles] = each_idx
                            break

        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump((valid_dict, test_dict), f)

    def load_output(self):
        with gzip.open(self.output().path, "rb") as f:
            valid_dict, test_dict = pickle.load(f)
        return valid_dict, test_dict


class Train(AutoNamingTask):
    output_ext = luigi.Parameter('pth')
    
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="train")

    def requires(self):
        return DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params)

    def run(self):
        hrg, prod_rule_seq_list = self.requires().load_output()
        prod_rule_seq_list_train = prod_rule_seq_list[: self.Train_params['num_train']]
        class_weight = None
        max_len = -1
        for each_prod_rule_seq in prod_rule_seq_list:
            if max_len < len(each_prod_rule_seq):
                max_len = len(each_prod_rule_seq)
        logger.info(f'max_len = {max_len}')
        min_val_loss = np.inf
        best_seed = None
        for each_seed in self.Train_params['seed_list']:
            torch.manual_seed(each_seed)
            hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
                = get_dataloaders(hrg, prod_rule_seq_list, self.Train_params)
            model = ae_catalog[self.Train_params['model']](
                hrg=hrg, class_weight=class_weight,
                **self.Train_params['model_params'], use_gpu=self.use_gpu)
            if self.use_gpu:
                model.cuda()
            train_loss, val_loss = model.fit(hrg_dataloader_train,
                                             data_loader_val=hrg_dataloader_val,
                                             max_num_examples=self.Train_params['num_early_stop'],
                                             print_freq=100,
                                             num_epochs=self.Train_params['num_epochs'],
                                             sgd=sgd_catalog[self.Train_params['sgd']],
                                             sgd_kwargs=self.Train_params['sgd_params'],
                                             logger=logger.info)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_seed = each_seed
        logger.info(f'best_seed = {best_seed}\tval_loss = {min_val_loss}')

        torch.manual_seed(best_seed)
        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list, self.Train_params)
        model = ae_catalog[self.Train_params['model']](
            hrg=hrg, class_weight=class_weight,
            **self.Train_params['model_params'], use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        train_loss, val_loss = model.fit(hrg_dataloader_train,
                                         data_loader_val=hrg_dataloader_val,
                                         print_freq=100,
                                         num_epochs=self.Train_params['num_epochs'],
                                         sgd=sgd_catalog[self.Train_params['sgd']],
                                         sgd_kwargs=self.Train_params['sgd_params'],
                                         logger=logger.info)
        torch.save((model.state_dict(), best_seed), self.output().path)

    def load_output(self):
        state_dict, seed = torch.load(self.output().path)
        return state_dict, seed


class Sample(MainTask, AutoNamingTask):
    output_ext = luigi.Parameter(default='txt')
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    Sample_params = luigi.DictParameter(default=Sample_params)
    use_gpu = luigi.BoolParameter()
    num_sampling = luigi.IntParameter(default=10)
    working_subdir = luigi.Parameter(default="sample")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()
        torch.manual_seed(seed)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = 100

        model = ae_catalog[self.Train_params['model']](
            hrg=hrg,  **model_params, use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        num_valid_mol = 0
        num_trials = 0
        for _ in range(self.num_sampling):
            hg_list = model.sample(sample_size=100)
            num_trials += 100
            for each_idx, each_hg in enumerate(hg_list):
                if each_hg is not None:
                    each_mol = hg_to_mol(each_hg)
                    each_smiles = Chem.MolToSmiles(each_mol)
                    mol_tmp = Chem.MolFromSmiles(each_smiles)
                    if mol_tmp is not None:
                        num_valid_mol += 1
                        with open(self.output().path + '.smi', 'a') as f:
                            f.write(each_smiles + '\n')
                    else:
                        logger.info(each_smiles)
        with open(self.output().path, 'w') as f:
            f.write(f'#(valid mol) = {num_valid_mol}, #(trials) = {num_trials}')

    def load_output(self):
        pass


class SampleWithPred(Sample):
    Train_params = luigi.DictParameter(default=TrainWithPred_params)
    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu)]


class CheckReconstructionRate(MainTask, AutoNamingTask):

    '''
    This task calculates the reconstruction error rate and validity rate.
    '''

    output_ext = luigi.Parameter(default='txt')
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    CheckReconstructionRate_params = luigi.DictParameter(default=CheckReconstructionRate_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="reconstruction_rate")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()
        torch.manual_seed(seed)

        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg,
                              prod_rule_seq_list,
                              self.Train_params,
                              batch_size=self.CheckReconstructionRate_params['batch_size'],
                              shuffle=False)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.CheckReconstructionRate_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg,  **model_params, use_gpu=self.use_gpu,
                                                       no_dropout=False)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        incorrect_mol_list = []
        num_mol_list = []
        num_success_list = []
        num_valid_list = []
        for _ in range(100):
            for each_dataloader in [hrg_dataloader_test]: #[hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test]:
                num_mol = 0
                num_success = 0
                num_valid = 0
                for each_batch in each_dataloader:
                    each_batch, num_pad = batch_padding(each_batch,
                                                        self.CheckReconstructionRate_params['batch_size'],
                                                        self.Train_params['model_params']['padding_idx'])
                    in_batch, _ = each_batch
                    in_batch = torch.LongTensor(np.mod(in_batch, model.vocab_size))
                    model.init_hidden()
                    in_batch_var = Variable(in_batch, requires_grad=False)
                    if self.use_gpu:
                        in_batch_var = in_batch_var.cuda()

                    mu, logvar = model.encode(in_batch_var)
                    z = model.reparameterize(mu, logvar)
                    if self.use_gpu:
                        torch.cuda.empty_cache()

                    model.init_hidden()
                    _, hg_list = model.decode(z=z,
                                              deterministic=self.CheckReconstructionRate_params['deterministic'],
                                              return_hg_list=True)
                    if num_pad != 0:
                        hg_list = hg_list[:-num_pad]
                    model.init_hidden()
                    if self.use_gpu:
                        torch.cuda.empty_cache()

                    for each_idx, each_hg in enumerate(hg_list):
                        num_mol += 1
                        try:
                            reconstructed_mol, is_stereo = hg_to_mol(each_hg, True)
                            num_valid += 1
                        except:
                            logger.warning(traceback.format_exc())
                            continue
                        if not is_stereo:
                            logger.info(f' stereo fails: {Chem.MolToSmiles(org_mol)}\t{Chem.MolToSmiles(reconstructed_mol)}')
                        len_seq = len(each_batch[0][each_idx]) - (each_batch[0][each_idx] == -1).sum()
                        org_mol = hg_to_mol(hrg.construct(each_batch[1][each_idx][:len_seq]))
                        if org_mol.HasSubstructMatch(reconstructed_mol, useChirality=True) \
                           and reconstructed_mol.HasSubstructMatch(org_mol, useChirality=True):
                            num_success += 1
                        else:
                            #incorrect_mol_list.append((org_mol, reconstructed_mol))
                            pass
                    
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                logger.info(f'#(mol) = {num_mol}, #(success) = {num_success}, #(valid) = {num_valid}')
                num_mol_list.append(num_mol)
                num_success_list.append(num_success)
                num_valid_list.append(num_valid)

            with open(self.output().path, 'a') as f:
                f.write(f'#(mol) = {num_mol}, #(reconstruction success) = {num_success}, #(valid) = {num_valid}\n')

        with open(self.output().path, 'a') as f:
            f.write('============ summary ============\n')
            f.write(f'#(mol) = {np.mean(num_mol_list)} +/- {np.std(num_mol_list)}, #(reconstruction success) = {np.mean(num_success_list)} +/- {np.std(num_success_list)}, #(valid) = {np.mean(num_valid_list)} +/- {np.std(num_valid_list)}\n')

    def load_output(self):
        pass


class CheckReconstructionRateWithPred(CheckReconstructionRate):

    '''
    This task calculates the reconstruction error rate and validity rate.
    '''

    Train_params = luigi.DictParameter(default=TrainWithPred_params)

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu)]


class Encode(AutoNamingTask):

    '''
    This task encodes the input dataset into continuous representations
    '''

    output_ext = luigi.Parameter(default='pklz')
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    Encode_params = luigi.DictParameter(default=Encode_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default='encode')

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()
        torch.manual_seed(seed)

        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg,
                              prod_rule_seq_list,
                              self.Train_params,
                              batch_size=self.Encode_params['batch_size'],
                              shuffle=False)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.Encode_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg,  **model_params, use_gpu=self.use_gpu,
                                                       no_dropout=False)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        embedding_list = []
        for each_dataloader in [hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test]:
            for each_batch in each_dataloader:
                each_batch, num_pad = batch_padding(each_batch,
                                                    self.Encode_params['batch_size'],
                                                    self.Train_params['model_params']['padding_idx'])
                in_batch, _ = each_batch
                in_batch = torch.LongTensor(np.mod(in_batch, model.vocab_size))
                model.init_hidden()
                in_batch_var = Variable(in_batch, requires_grad=False)
                if self.use_gpu:
                    in_batch_var = in_batch_var.cuda()

                mu, _ = model.encode(in_batch_var)
                if num_pad:
                    mu = mu[:-num_pad]
                embedding_list.append(mu.cpu().detach().numpy())

                if self.use_gpu:
                    torch.cuda.empty_cache()
                model.init_hidden()

        embedding = np.vstack(embedding_list)
        logger.info(f'embedding.shape = {embedding.shape}')
        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump(embedding, f)

    def load_output(self):
        with gzip.open(self.output().path, 'rb') as f:
            X = pickle.load(f)
        return X

class EncodeWithPred(Encode):

    '''
    This task encodes the input dataset into continuous representations
    '''

    Train_params = luigi.DictParameter(default=TrainWithPred_params)

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu)]


class ComputeTargetValues(AutoNamingTask):

    '''
    This task computes target values.
    ** only training set is used, which will be randomly split into train and test sets in the BO setting. **
    '''

    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    working_subdir = luigi.Parameter(default="target_values")

    def requires(self):
        return []

    def run(self):
        if self.ComputeTargetValues_params['target'] == 'logP - SA - cycle':
            log_p_list = []
            sa_list = []
            cycle_score_list = []

            mol_gen = Chem.SmilesMolSupplier(os.path.join("INPUT", 'data', "train_val_test.txt"), titleLine=False)
            sa_list = synthetic_accessibility_batch(mol_gen, logger=logger.info)
            mol_gen = Chem.SmilesMolSupplier(os.path.join("INPUT", 'data', "train_val_test.txt"), titleLine=False)
            for each_mol in mol_gen:
                log_p_list.append(log_p(each_mol))
                #sa_list.append(synthetic_accessibility(each_mol))
                cycle_score_list.append(cycle_score(each_mol))
            log_p_normalized = (np.array(log_p_list) - np.mean(log_p_list[:self.ComputeTargetValues_params['num_train']])) \
                               / np.std(log_p_list[:self.ComputeTargetValues_params['num_train']])
            sa_normalized = (np.array(sa_list) - np.mean(sa_list[: self.ComputeTargetValues_params['num_train']])) \
                            / np.std(sa_list[:self.ComputeTargetValues_params['num_train']])
            cycle_score_normalized = (np.array(cycle_score_list) - np.mean(cycle_score_list[:self.ComputeTargetValues_params['num_train']])) \
                                     / np.std(cycle_score_list[:self.ComputeTargetValues_params['num_train']])

            y_all = log_p_normalized - sa_normalized - cycle_score_normalized
            y_all = - y_all # y should be minimized
            y_raw = {'log_p_list': log_p_list, 'sa_list': sa_list, 'cycle_score_list': cycle_score_list}
            with gzip.open(self.output().path, 'wb') as f:
                pickle.dump((y_all, y_raw), f)
        elif self.ComputeTargetValues_params['target'] == 'logP - SA':
            # not normalized
            log_p_list = []
            sa_list = []

            mol_gen = Chem.SmilesMolSupplier(os.path.join("INPUT", 'data', "train_val_test.txt"), titleLine=False)
            sa_list = synthetic_accessibility_batch(mol_gen, logger=logger.info)
            mol_gen = Chem.SmilesMolSupplier(os.path.join("INPUT", 'data', "train_val_test.txt"), titleLine=False)
            for each_mol in mol_gen:
                log_p_list.append(log_p(each_mol))

            y_all = np.array(log_p_list) - np.array(sa_list)
            y_all = - y_all # y should be minimized
            y_raw = {'log_p_list': log_p_list, 'sa_list': sa_list}
            with gzip.open(self.output().path, 'wb') as f:
                pickle.dump((y_all, y_raw), f)

    def load_output(self):
        with gzip.open(self.output().path, 'rb') as f:
            y_all, y_raw = pickle.load(f)
        return y_all, y_raw


class ConstructDatasetForBO(AutoNamingTask):

    '''
    This task constructs a dataset for Bayesian optimization.
    ** only training set is used, which will be randomly split into train and test sets in the BO setting. **
    '''

    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    ConstructDatasetForBO_params = luigi.DictParameter(default=ConstructDatasetForBO_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="bo_dataset")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()

        torch.manual_seed(seed)
        np.random.seed(self.ConstructDatasetForBO_params['seed'])
        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list, self.Train_params,
                              batch_size=self.ConstructDatasetForBO_params['batch_size'],
                              shuffle=False)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.ConstructDatasetForBO_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg, **model_params,
                                                       use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        latent_vector_list = []
        for each_dataloader in [hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test]:
            for each_batch in each_dataloader:
                each_batch, num_pad = batch_padding(each_batch,
                                                    self.ConstructDatasetForBO_params['batch_size'],
                                                    self.Train_params['model_params']['padding_idx'])
                in_batch, _ = each_batch
                in_batch = torch.LongTensor(np.mod(in_batch, model.vocab_size))
                model.init_hidden()
                in_batch_var = Variable(in_batch, requires_grad=False)
                if self.use_gpu:
                    in_batch_var = in_batch_var.cuda()

                mu, logvar = model.encode(in_batch_var)
                z = model.reparameterize(mu, logvar, False)
                z = z.cpu().cpu().detach().numpy()
                if num_pad:
                    z = z[:-num_pad]
                latent_vector_list.append(z)
        X_all = np.concatenate(latent_vector_list, 0)

        # target val
        y_all, y_raw = self.requires()[2].load_output()

        assert X_all.shape[0] == len(y_all.ravel()), 'X_all and y_all have inconsistent shapes'
        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump((X_all, y_all, y_raw), f)

    def load_output(self):
        with gzip.open(self.output().path, 'rb') as f:
            X_all, y_all, y_raw = pickle.load(f)
        return X_all, y_all, y_raw


class ConstructDatasetForBOWithPred(ConstructDatasetForBO):

    '''
    This task constructs a dataset for Bayesian optimization.
    ** only training set is used, which will be randomly split into train and test sets in the BO setting. **
    '''

    Train_params = luigi.DictParameter(default=TrainWithPred_params)

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params)]


class BayesianOptimization(MainTask, AutoNamingTask):

    '''
    This task calculates the reconstruction error rate and validity rate.
    '''

    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    ConstructDatasetForBO_params = luigi.DictParameter(default=ConstructDatasetForBO_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    BayesianOptimization_params = luigi.DictParameter(default=BayesianOptimization_params)
    use_gpu = luigi.BoolParameter()
    seed = luigi.IntParameter(default=123)
    working_subdir = luigi.Parameter(default="bayesian_optimization")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu),
                ConstructDatasetForBO(
                    DataPreprocessing_params=self.DataPreprocessing_params,
                    Train_params=self.Train_params,
                    ConstructDatasetForBO_params=self.ConstructDatasetForBO_params,
                    ComputeTargetValues_params=self.ComputeTargetValues_params,
                    use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()

        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list,
                              self.Train_params,
                              batch_size=self.ConstructDatasetForBO_params['batch_size'],
                              shuffle=False)
        torch.manual_seed(seed)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.ConstructDatasetForBO_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg,  **model_params,
                                                       use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        X_all, y_all, y_raw = self.requires()[2].load_output()
        log_p_mean = np.mean(y_raw['log_p_list'][:self.ComputeTargetValues_params['num_train']])
        log_p_std = np.std(y_raw['log_p_list'][:self.ComputeTargetValues_params['num_train']])
        sa_mean = np.mean(y_raw['sa_list'][:self.ComputeTargetValues_params['num_train']])
        sa_std = np.std(y_raw['sa_list'][:self.ComputeTargetValues_params['num_train']])
        cycle_score_mean = np.mean(
            y_raw['cycle_score_list'][:self.ComputeTargetValues_params['num_train']])
        cycle_score_std = np.std(
            y_raw['cycle_score_list'][:self.ComputeTargetValues_params['num_train']])

        def target_func(mol):
            ''' target function should be minimized.
            '''
            tgt = ((log_p(mol) - log_p_mean) / log_p_std) \
                  - ((synthetic_accessibility(mol) - sa_mean) / sa_std) \
                  - ((cycle_score(mol) - cycle_score_mean) / cycle_score_std)
            return -tgt

        assert round(-target_func(Chem.MolFromSmiles(
            'ClC1=CC=C2C(C=C(C('
            'C)=O)C(C(NC3=CC(NC('
            'NC4=CC(C5=C('
            'C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1')), 2) == 5.30

        X = X_all[:self.ComputeTargetValues_params['num_train']]
        y = y_all[:self.ComputeTargetValues_params['num_train']]
        X_test = X_all[self.ComputeTargetValues_params['num_train']:]
        y_test = y_all[self.ComputeTargetValues_params['num_train']:]

        mol_opt = MolecularOptimization(target_func, model, seed=self.seed)
        num_mol = X.shape[0]
        np.random.seed(self.seed)
        permutation = np.random.choice(num_mol, num_mol, replace=False)

        X_train = X[permutation, :][0:np.int(np.round(0.9 * num_mol)), :]
        #X_test = X[permutation, :][np.int(np.round(0.9 * num_mol)):, :]

        y_train = y[permutation][0: np.int(np.round(0.9 * num_mol))]
        #y_test = y[permutation][np.int(np.round(0.9 * num_mol)):]

        history = mol_opt.run(X_train, y_train,
                              X_test, y_test,
                              logger=logger.info,
                              **self.BayesianOptimization_params['run_params'])

        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump(history, f)

    def load_output(self):
        with gzip.open(self.output().path, 'rb') as f:
            history = pickle.load(f)
        return history



class BayesianOptimizationWithPred(BayesianOptimization):

    '''
    This task calculates the reconstruction error rate and validity rate.
    '''

    Train_params = luigi.DictParameter(default=TrainWithPred_params)

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu),
                ConstructDatasetForBOWithPred(
                    DataPreprocessing_params=self.DataPreprocessing_params,
                    Train_params=self.Train_params,
                    ConstructDatasetForBO_params=self.ConstructDatasetForBO_params,
                    ComputeTargetValues_params=self.ComputeTargetValues_params,
                    use_gpu=self.use_gpu)]


class GuacaMolGoalDirectedBenchmarkSuite(MainTask, AutoNamingTask):

    '''
    Run Guacamol benchmark tests
    '''
    output_ext = luigi.Parameter('json')

    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    Encode_params = luigi.DictParameter(default=Encode_params)
    GuacaMol_params = luigi.DictParameter(default=GuacaMol_params)
    use_gpu = luigi.BoolParameter()
    seed = luigi.IntParameter(default=123)
    working_subdir = luigi.Parameter(default="guacamol_goal")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu),
                Encode(DataPreprocessing_params=self.DataPreprocessing_params,
                       Train_params=self.Train_params,
                       use_gpu=self.use_gpu,
                       Encode_params=self.Encode_params)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()

        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list,
                              self.Train_params,
                              batch_size=self.GuacaMol_params['batch_size'],
                              shuffle=False)
        torch.manual_seed(seed)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.GuacaMol_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg,  **model_params,
                                                       use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        X = self.requires()[2].load_output()

        # select train data set
        X = X[:self.Train_params['num_train']]
        goal_directed_gen = MHGDirectedGenerator(
            model,
            X,
            os.path.join("INPUT", 'data', "train_val_test.txt"),
            self.Train_params['num_train'],
            GuacaMol_params['bo_run_params'],
            self.seed,
            logger=logger.info)

        from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
        assess_goal_directed_generation(goal_directed_gen,
                                        json_output_file=self.output().path,
                                        benchmark_version=self.GuacaMol_params['suite'])


class GuacaMolDistributionMatchingBenchmarkSuite(MainTask, AutoNamingTask):

    '''
    Run Guacamol benchmark tests
    '''
    output_ext = luigi.Parameter('json')

    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    GuacaMol_params = luigi.DictParameter(default=GuacaMol_params)
    use_gpu = luigi.BoolParameter()
    seed = luigi.IntParameter(default=123)
    working_subdir = luigi.Parameter(default="guacamol_dist")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()

        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list,
                              self.Train_params,
                              batch_size=self.GuacaMol_params['batch_size'],
                              shuffle=False)
        torch.manual_seed(seed)
        model_params = deepcopy(dict(self.Train_params['model_params']))
        model_params['batch_size'] = self.GuacaMol_params['batch_size']
        model = ae_catalog[self.Train_params['model']](hrg=hrg,  **model_params,
                                                       use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)
        torch.no_grad()

        optimizer = MHGDistributionMatchingGenerator(model, deterministic=self.GuacaMol_params['deterministic'])

        from guacamol.assess_distribution_learning import assess_distribution_learning
        from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
        assess_distribution_learning(optimizer,
                                     chembl_training_file=os.path.join('INPUT', 'data', 'train_val_test.txt'),
                                     json_output_file=self.output().path,
                                     benchmark_version=self.GuacaMol_params['suite'])


class MultipleBayesianOptimization(MainTask, AutoNamingTask):

    '''
    This task calculates the reconstruction error rate and validity rate.
    '''

    output_ext = luigi.Parameter(default='txt')
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    ConstructDatasetForBO_params = luigi.DictParameter(default=ConstructDatasetForBO_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    BayesianOptimization_params = luigi.DictParameter(default=BayesianOptimization_params)
    MultipleBayesianOptimization_params = luigi.DictParameter(default=MultipleBayesianOptimization_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="multiple_bayesian_optimization")

    def requires(self):
        return [BayesianOptimization(DataPreprocessing_params=self.DataPreprocessing_params,
                                     Train_params=self.Train_params,
                                     BayesianOptimization_params=self.BayesianOptimization_params,
                                     ConstructDatasetForBO_params=self.ConstructDatasetForBO_params,
                                     ComputeTargetValues_params=self.ComputeTargetValues_params,
                                     use_gpu=self.use_gpu,
                                     working_dir=self.working_dir,
                                     seed=each_seed) for each_seed in self.MultipleBayesianOptimization_params['seed_list']]

    def run(self):
        history_list = []
        for each_task in self.requires():
            history_list.append(each_task.load_output())
        '''
        for each_file in self.input():
            with gzip.open(each_file.path, 'rb') as f:
                history_list.append(pickle.load(f))
        '''
        num_trials = len(history_list)
        train_err_list = []
        train_ll_list = []
        test_err_list = []
        test_ll_list = []
        mol_score_list = []
        for each_history in history_list:
            for each_idx in range(len(each_history)):
                train_err_list.append(each_history[each_idx]['train_err'])
                train_ll_list.append(each_history[each_idx]['train_ll'])
                test_err_list.append(each_history[each_idx]['test_err'])
                test_ll_list.append(each_history[each_idx]['test_ll'])
                mol_score_list.extend(list(zip(each_history[each_idx]['mol_list'],
                                               each_history[each_idx]['score_list'])))

        # In BO, scores are minimized.
        # below, scores are presented using the correct sign, the larger the better.
        mol_score_list = [(Chem.MolToSmiles(x), -y) for x, y in mol_score_list]
        mol_score_list = list(set(mol_score_list))
        mol_score_list = sorted(mol_score_list, key=lambda x: x[1], reverse=True)

        with open(self.output().path, 'w') as f:
            f.write(f'train_err = {np.mean(train_err_list)} +/- {np.std(train_err_list)}\n')
            f.write(f'train_ll = {np.mean(train_ll_list)} +/- {np.std(train_ll_list)}\n')
            f.write(f'test_err = {np.mean(test_err_list)} +/- {np.std(test_err_list)}\n')
            f.write(f'test_ll = {np.mean(test_ll_list)} +/- {np.std(test_ll_list)}\n')
            for mol, score in mol_score_list[:50]:
                f.write(f'{score}\t{mol}\n')

class MultipleBayesianOptimizationWithPred(MultipleBayesianOptimization):

    Train_params = luigi.DictParameter(default=TrainWithPred_params)

    def requires(self):
        return [BayesianOptimizationWithPred(
            DataPreprocessing_params=self.DataPreprocessing_params,
            Train_params=self.Train_params,
            BayesianOptimization_params=self.BayesianOptimization_params,
            ConstructDatasetForBO_params=self.ConstructDatasetForBO_params,
            ComputeTargetValues_params=self.ComputeTargetValues_params,
            use_gpu=self.use_gpu,
            working_dir=self.working_dir,
            seed=each_seed) for each_seed in self.MultipleBayesianOptimization_params['seed_list']]

    
class TrainWithPred(AutoNamingTask):
    output_ext = luigi.Parameter('pth')
    
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    TrainWithPred_params = luigi.DictParameter(default=TrainWithPred_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="train_with_pred")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        y_all, y_raw = self.requires()[1].load_output()
        prod_rule_seq_list_train = prod_rule_seq_list[: self.TrainWithPred_params['num_train']]
        class_weight = None
        min_val_loss = np.inf
        best_seed = None
        for each_seed in self.TrainWithPred_params['seed_list']:
            torch.manual_seed(each_seed)
            np.random.seed(each_seed)
            hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
                = get_dataloaders(hrg, prod_rule_seq_list,
                                  self.TrainWithPred_params, target_val_list=y_all.ravel())
            model = ae_catalog[self.TrainWithPred_params['model']](
                hrg=hrg, class_weight=class_weight,
                **self.TrainWithPred_params['model_params'], use_gpu=self.use_gpu)
            if self.use_gpu:
                model.cuda()
            train_loss, val_loss = model.fit(hrg_dataloader_train,
                                             data_loader_val=hrg_dataloader_val,
                                             max_num_examples=self.TrainWithPred_params['num_early_stop'],
                                             print_freq=100,
                                             num_epochs=self.TrainWithPred_params['num_epochs'],
                                             sgd=sgd_catalog[self.TrainWithPred_params['sgd']],
                                             sgd_kwargs=self.TrainWithPred_params['sgd_params'],
                                             logger=logger.info)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_seed = each_seed
        logger.info(f'best_seed = {best_seed}\tval_loss = {min_val_loss}')

        torch.manual_seed(best_seed)
        np.random.seed(best_seed)
        hrg_dataloader_train, hrg_dataloader_val, hrg_dataloader_test \
            = get_dataloaders(hrg, prod_rule_seq_list, self.TrainWithPred_params,
                              target_val_list=y_all.ravel())
        model = ae_catalog[self.TrainWithPred_params['model']](
            hrg=hrg, class_weight=class_weight,
            **self.TrainWithPred_params['model_params'], use_gpu=self.use_gpu)
        if self.use_gpu:
            model.cuda()
        train_loss, val_loss = model.fit(hrg_dataloader_train,
                                         data_loader_val=hrg_dataloader_val,
                                         print_freq=100,
                                         num_epochs=self.TrainWithPred_params['num_epochs'],
                                         sgd=sgd_catalog[self.TrainWithPred_params['sgd']],
                                         sgd_kwargs=self.TrainWithPred_params['sgd_params'],
                                         logger=logger.info)
        torch.save((model.state_dict(), best_seed), self.output().path)

    def load_output(self):
        state_dict, seed = torch.load(self.output().path)
        return state_dict, seed


class ConstrainedMolOpt(AutoNamingTask, MainTask):
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    TrainWithPred_params = luigi.DictParameter(default=TrainWithPred_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    ConstrainedMolOpt_params = luigi.DictParameter(default=ConstrainedMolOpt_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="constrained_mol_opt_with_pred")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                DataPreprocessing4ConstrainedMolOpt(
                    DataPreprocessing_params=self.DataPreprocessing_params),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              ComputeTargetValues_params=self.ComputeTargetValues_params,
                              TrainWithPred_params=self.TrainWithPred_params,
                              use_gpu=self.use_gpu)]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        valid_dict, test_dict = self.requires()[1].load_output()
        y_all, y_raw = self.requires()[2].load_output()
        state_dict, seed = self.requires()[3].load_output()

        torch.manual_seed(seed)
        np.random.seed(self.ConstrainedMolOpt_params['seed'])
        model_params = deepcopy(dict(self.TrainWithPred_params['model_params']))
        model_params['batch_size'] = 1
        model = ae_catalog[self.TrainWithPred_params['model']](
            hrg=hrg,  **model_params,
            use_gpu=self.use_gpu, no_dropout=True)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)

        _fscores = load_fscores()
        def target_func(mol, _fscores):
            ''' target function should be minimized.
            '''
            tgt = log_p(mol) - synthetic_accessibility(mol, _fscores)
            return -tgt # the lower, the better

        res = pd.DataFrame(columns=['org_smiles', 'mod_smiles',
                                    'org_tgt', 'mod_tgt',
                                    'improvement',
                                    'similarity'])

        prod_rule_seq_list_test = []
        smiles_list = []
        for each_smiles, each_idx in test_dict.items():
            prod_rule_seq_list_test.append(prod_rule_seq_list[each_idx])
            smiles_list.append(each_smiles)
        hrg_dataset = HRGDataset(hrg,
                                 prod_rule_seq_list_test,
                                 self.TrainWithPred_params['model_params']['max_len'],
                                 inversed_input=self.TrainWithPred_params['inversed_input'])
        hrg_dataloader = DataLoader(dataset=hrg_dataset,
                                    batch_size=1,
                                    shuffle=False, drop_last=False)

        res_idx = 0
        for each_idx, each_in_seq in enumerate(hrg_dataloader):
            if each_idx % 50 == 0:
                logger.info(f'{each_idx}/{len(hrg_dataloader)}')
            in_seq = each_in_seq[0]
            each_smiles = smiles_list[each_idx]
            org_mol = Chem.MolFromSmiles(each_smiles)
            org_tgt = target_func(Chem.MolFromSmiles(each_smiles), _fscores)
            org_mol_fp = AllChem.GetMorganFingerprint(
                Chem.MolFromSmiles(each_smiles), 2)

            # optimize
            hg_list, pred_list, z_trajectory = model.inseq_optimization(
                in_seq, maximize=False, **self.ConstrainedMolOpt_params['inseq_opt_params'])
            candidate_smiles_list = []
            candidate_fp_list = []
            similarity_list = []
            for each_hg_idx, each_hg in enumerate(hg_list):
                candidate_df = pd.DataFrame(columns=res.columns,
                                            index=[res_idx])
                candidate_df.loc[res_idx, 'org_smiles'] = each_smiles
                candidate_df.loc[res_idx, 'org_tgt'] = org_tgt

                try:
                    mod_mol = hg_to_mol(each_hg)
                    mod_smiles = Chem.MolToSmiles(mod_mol)
                    candidate_df.loc[res_idx, 'mod_smiles'] = mod_smiles
                    candidate_df.loc[res_idx, 'success'] \
                        = not(org_mol.HasSubstructMatch(mod_mol, useChirality=True) \
                              and mod_mol.HasSubstructMatch(org_mol, useChirality=True))
                    candidate_fp = AllChem.GetMorganFingerprint(mod_mol, 2)
                    candidate_df.loc[res_idx, 'similarity'] \
                        = DataStructs.TanimotoSimilarity(org_mol_fp, candidate_fp)
                    candidate_df.loc[res_idx, 'mod_tgt'] = target_func(mod_mol, _fscores)
                    candidate_df.loc[res_idx, 'pred_mod_tgt'] = float(pred_list[each_hg_idx])
                    candidate_df.loc[res_idx, 'improvement'] \
                        = candidate_df.loc[res_idx, 'mod_tgt'] - candidate_df.loc[res_idx, 'org_tgt']
                except:
                    candidate_df.loc[res_idx, 'mod_smiles'] = ''
                    candidate_df.loc[res_idx, 'success'] = False
                    candidate_df.loc[res_idx, 'similarity'] = -1
                    candidate_df.loc[res_idx, 'mod_tgt'] = np.inf
                    candidate_df.loc[res_idx, 'pred_mod_tgt'] = np.inf
                    candidate_df.loc[res_idx, 'improvement'] = np.inf
                res = res.append(candidate_df)
                res_idx += 1

        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump(res, f)

    def load_output(self):
        with gzip.open(self.output().path, 'rb') as f:
            res = pickle.load(f)
        return res


class SummaryConstrainedMolOpt(MainTask, AutoNamingTask):
    '''
    Summarize the results of the constrained molecular optimization task.
    The data provided from the previous task use the raw 
    '''
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    TrainWithPred_params = luigi.DictParameter(default=TrainWithPred_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    ConstrainedMolOpt_params = luigi.DictParameter(default=ConstrainedMolOpt_params)
    use_gpu = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="summary_constrained_molopt_with_pred")

    def requires(self):
        return ConstrainedMolOpt(DataPreprocessing_params=self.DataPreprocessing_params,
                                 TrainWithPred_params=self.TrainWithPred_params,
                                 ComputeTargetValues_params=self.ComputeTargetValues_params,
                                 ConstrainedMolOpt_params=self.ConstrainedMolOpt_params,
                                 use_gpu=self.use_gpu,
                                 working_dir=self.working_dir)

    def run(self):
        res = self.requires().load_output()
        org_smiles_list = list(set(res['org_smiles']))
        similarity_list = [0, 0.2, 0.4, 0.6]
        res_dict = {}
        logger.info('============================================================')
        logger.info(f'improvement:\tthe lower, the better')
        logger.info(f'similarity:\tthe higher, the better')
        logger.info(f'success ratio:\tthe higher, the better')
        logger.info('============================================================')

        for each_similarity in similarity_list:
            summary_df = pd.DataFrame(columns=res.columns)
            for each_org_smiles in org_smiles_list:
                batch_res = res.loc[res['org_smiles'] == each_org_smiles]
                batch_res = batch_res[batch_res['similarity'] != 1]
                similar_res = batch_res[batch_res['similarity'] >= each_similarity]
                if len(similar_res) == 0:
                    continue
                else:
                    best_idx = similar_res['pred_mod_tgt'].argmin()
                    summary_df = summary_df.append(similar_res.loc[best_idx])
            res_dict[each_similarity] = deepcopy(summary_df)
            logger.info('============================================================')
            logger.info(f'similarity threshold:\t{each_similarity}')
            logger.info(f'improvement:\t{summary_df["improvement"].mean()} +/- {summary_df["improvement"].std()}')
            logger.info(f'similarity:\t{summary_df["similarity"].mean()} +/- {summary_df["similarity"].std()}')
            logger.info(f'success ratio: {len(summary_df)/len(org_smiles_list)}')
            logger.info('============================================================')

        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump(res_dict, f)

class RandomSearch(MainTask, AutoNamingTask):
    DataPreprocessing_params = luigi.DictParameter(default=DataPreprocessing_params)
    Train_params = luigi.DictParameter(default=Train_params)
    ComputeTargetValues_params = luigi.DictParameter(default=ComputeTargetValues_params)
    num_batch = luigi.IntParameter(default=100)
    use_gpu = luigi.BoolParameter()
    draw = luigi.BoolParameter()
    working_subdir = luigi.Parameter(default="random_search")

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                Train(DataPreprocessing_params=self.DataPreprocessing_params,
                      Train_params=self.Train_params, use_gpu=self.use_gpu),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params)
        ]

    def run(self):
        hrg, prod_rule_seq_list = self.requires()[0].load_output()
        state_dict, seed = self.requires()[1].load_output()

        torch.manual_seed(seed)
        model = ae_catalog[self.Train_params['model']](
            hrg=hrg,  **self.Train_params['model_params'], use_gpu=self.use_gpu, no_dropout=True)
        if self.use_gpu:
            model.cuda()
        model.load_state_dict(state_dict)

        y_all, y_raw = self.requires()[2].load_output()
        log_p_mean = np.mean(y_raw['log_p_list'][:self.ComputeTargetValues_params['num_train']])
        log_p_std = np.std(y_raw['log_p_list'][:self.ComputeTargetValues_params['num_train']])
        sa_mean = np.mean(y_raw['sa_list'][:self.ComputeTargetValues_params['num_train']])
        sa_std = np.std(y_raw['sa_list'][:self.ComputeTargetValues_params['num_train']])
        cycle_score_mean = np.mean(y_raw['cycle_score_list'][:self.ComputeTargetValues_params['num_train']])
        cycle_score_std = np.std(y_raw['cycle_score_list'][:self.ComputeTargetValues_params['num_train']])

        def target_func(mol):
            ''' target function should be minimized.
            '''
            tgt = ((log_p(mol) - log_p_mean) / log_p_std) \
                  - ((synthetic_accessibility(mol) - sa_mean) / sa_std) \
                  - ((cycle_score(mol) - cycle_score_mean) / cycle_score_std)
            return -tgt

        def target_func_batch(mol_list):
            ''' target function should be minimized.
            '''
            sa_list = synthetic_accessibility_batch(mol_list)
            tgt_list = []
            for each_idx, each_mol in enumerate(mol_list):
                tgt = ((log_p(each_mol) - log_p_mean) / log_p_std) \
                      - ((sa_list[each_idx] - sa_mean) / sa_std) \
                      - ((cycle_score(each_mol) - cycle_score_mean) / cycle_score_std)
                tgt_list.append(-tgt)
            return tgt_list

        sampled_mol_list = []
        min_mol = None
        min_tgt = np.inf
        for each_batch in range(self.num_batch):
            if self.Train_params['model'] == 'GrammarSeq2SeqVAEWithPred':
                hg_list, tgt_pred_list, z = model.sample(return_z_pred=True)
            else:
                hg_list, z = model.sample(return_z=True)
            mol_list = []
            for each_hg in hg_list:
                try:
                    mol_list.append(Chem.MolFromSmiles(Chem.MolToSmiles(hg_to_mol(each_hg))))
                except:
                    logger.info('failed')
            tgt_list = target_func_batch(mol_list)
            for each_idx in range(len(tgt_list)):
                sampled_mol_list.append({'smiles': Chem.MolToSmiles(mol_list[each_idx]),
                                         'z': z[each_idx],
                                         'tgt': tgt_list[each_idx]})
                if tgt_list[each_idx] < -3.5:
                    logger.info(f'{tgt_list[each_idx]}\t{np.linalg.norm(z[each_idx])}\t{Chem.MolToSmiles(mol_list[each_idx])}')

        lipschitz_list = []
        from itertools import combinations
        for each_info_1, each_info_2 in combinations(sampled_mol_list, 2):
            lipschitz_list.append(np.abs(each_info_1['tgt'] - each_info_2['tgt'])\
                                  / np.linalg.norm(each_info_1['z'] - each_info_2['z']))
        logger.info(f'max={np.max(lipschitz_list)}, min={np.min(lipschitz_list)}, mean={np.mean(lipschitz_list)}')
        with gzip.open(self.output().path, 'wb') as f:
            pickle.dump((sampled_mol_list, lipschitz_list), f)


class RandomSearchWithPred(RandomSearch):
            
    Train_params = luigi.DictParameter(default=TrainWithPred_params)            

    def requires(self):
        return [DataPreprocessing(DataPreprocessing_params=self.DataPreprocessing_params),
                TrainWithPred(DataPreprocessing_params=self.DataPreprocessing_params,
                              TrainWithPred_params=self.Train_params, use_gpu=self.use_gpu),
                ComputeTargetValues(ComputeTargetValues_params=self.ComputeTargetValues_params)]


if __name__ == "__main__":
    for each_engine_status in glob.glob("./engine_status.*"):
        os.remove(each_engine_status)
    with open("engine_status.ready", "w") as f:
        f.write("ready: {}\n".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')))

    '''
    import cProfile
    cProfile.run('main()', filename='main.prof')
    import pstats
    stats = pstats.Stats('main.prof')
    stats.sort_stats('time')
    stats.print_stats()
    '''
    main()
