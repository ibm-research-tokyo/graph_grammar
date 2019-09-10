#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2019"
__version__ = "0.1"
__date__ = "May 15 2019"

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import MoleculewiseScoringFunction, BatchScoringFunction
from guacamol.assess_goal_directed_generation import _evaluate_goal_directed_benchmarks
from guacamol.score_modifier import LinearModifier
from guacamol.utils.helpers import setup_default_logger
from rdkit import Chem
from collections import OrderedDict
from ..descriptors.descriptors import log_p, load_fscores, synthetic_accessibility, cycle_score, synthetic_accessibility_batch
from ..bo.mol_opt import MolecularOptimization
from ..io.smi import hg_to_mol
import guacamol
import numpy as np
import argparse
import json
import os


class MHGDistributionMatchingGenerator(DistributionMatchingGenerator):

    ''' MHG-VAE for distribution matching generator.

    Attributes
    ----------
    model : GrammarSeq2SeqVAE
        trained model
    '''

    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def generate(self, number_samples):
        hg_list = self.model.sample(sample_size=number_samples, deterministic=self.deterministic)
        mol_list = [hg_to_mol(each_hg) for each_hg in hg_list]
        return [Chem.MolToSmiles(each_mol) for each_mol in mol_list]


class MHGDirectedGenerator(GoalDirectedGenerator):

    '''

    Attributes
    ----------
    model : 
        trained model
    '''

    def __init__(self, model, X, path_to_train_val_test_smiles,
                 num_train, bo_run_params, seed, logger=print):
        self.model = model
        self.X = X
        self.path_to_train_val_test_smiles = path_to_train_val_test_smiles
        self.num_train = num_train
        self.bo_run_params = bo_run_params
        self.seed = seed
        self.logger = logger

    def generate_optimized_molecules(self,
                                     scoring_function,
                                     number_molecules,
                                     starting_population=None):
        '''
        Parameters
        ----------
        scoring_function : ScoringFunction
            a scoring function to be maximized.
        num_mol : int
            the number of molecules to be outputed
        starting_smiles_list : list, optional
            list of smiles strings from which the optimization starts

        Returns
        -------
        list of smiles strings
            
        '''
        def _my_scoring_function(mol, my_scoring_function=scoring_function.score):
            try:
                smiles = Chem.MolToSmiles(mol)
            except:
                raise ValueError(f'{mol} is not valid.')
            if smiles is None:
                raise ValueError(f'{mol} is not valid.')
            else:
                return - my_scoring_function(smiles)

        def _my_scoring_function4smiles(smiles, my_scoring_function=scoring_function.score):
            return - my_scoring_function(smiles)

        # using training data only
        X = self.X
        y = self.get_target_prop(_my_scoring_function4smiles)
        assert X.shape[0] == len(y)

        mol_opt = MolecularOptimization(_my_scoring_function, self.model, seed=self.seed)
        arg_idx = np.argsort(y)

        X_train = X[arg_idx[:self.bo_run_params['num_train']]]
        X_test = None
        y_train = y[arg_idx[:self.bo_run_params['num_train']]]
        y_test = None
        history = mol_opt.run(X_train, y_train,
                              X_test, y_test,
                              logger=self.logger,
                              **self.bo_run_params)

        mol_score_list = []
        for each_idx in range(len(history)):
            mol_score_list.extend(list(zip(history[each_idx]['mol_list'],
                                           history[each_idx]['score_list'])))
        # In BO, scores are minimized.
        # below, scores are presented using the correct sign, the larger the better.
        mol_score_list = [(Chem.MolToSmiles(x), -y) for x, y in mol_score_list]
        mol_score_list = list(set(mol_score_list))
        mol_score_list = sorted(mol_score_list, key=lambda x: x[1], reverse=True)
        return_mol_list = [each_mol_score[0] for each_mol_score \
                           in mol_score_list[:number_molecules]]
        return return_mol_list

    def get_target_prop(self, scoring_function):
        y_list = []
        with open(self.path_to_train_val_test_smiles, 'r') as f:
            for each_smiles in f:
                y_list.append(scoring_function(each_smiles))
        return np.array(y_list[:self.num_train]).ravel()
