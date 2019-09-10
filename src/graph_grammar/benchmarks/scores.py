#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Benchmark tasks """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2019"
__version__ = "0.1"
__date__ = "Mar 15 2019"

from guacamol.scoring_function import MoleculewiseScoringFunction, BatchScoringFunction
from rdkit import Chem
from ..descriptors.descriptors import log_p, load_fscores, synthetic_accessibility, cycle_score, synthetic_accessibility_batch


class StandardizedPenalizedLogP(MoleculewiseScoringFunction):

    ''' Standardized penalized log P, which is calculated by
    log_p - synthetic_accessibility - cycle
    '''

    def raw_score(self, smiles):
        if not hasattr(self, 'fscores'):
            self.fscores = load_fscores()
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = 3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = 0.0485696876403053
        cycle_std = 0.2860212110245455
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return None

        if mol is None:
            return None
        else:
            standardized_logP = (log_p(mol) - logP_mean) / logP_std
            standardized_sa = (synthetic_accessibility(mol, self.fscores) - SA_mean) / SA_std
            standardized_cycle = (cycle_score(mol) - cycle_mean) / cycle_std
            score = standardized_logP - standardized_sa - standardized_cycle # to be maximized.
            return score


class BatchStandardizedPenalizedLogP(BatchScoringFunction):

    ''' Batch implementation of the standardized penalized log P
    '''

    def raw_score_list(self, smiles_list):
        ''' compute raw scores

        Parameters
        ----------
        smiles_list : list of strings
            each string corresponds to a SMILES representation of a molecule

        Returns
        -------
        score_list : list of floats
            each corresponds to the score of each molecule.
            if the score cannot be computed, then the element should be `None`.
        '''
        if not hasattr(self, 'fscores'):
            self.fscores = load_fscores()

        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = 3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = 0.0485696876403053
        cycle_std = 0.2860212110245455

        test_mol = Chem.MolFromSmiles('ClC1=CC=C2C(C=C(C('
                                      'C)=O)C(C(NC3=CC(NC('
                                      'NC4=CC(C5=C('
                                      'C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1')
        assert round((log_p(test_mol) - logP_mean)/logP_std \
                     - (cycle_score(test_mol) - cycle_mean)/cycle_std\
                     - (synthetic_accessibility(test_mol, self.fscores) - SA_mean)/SA_std, 2) == 5.30

        mol_list = []
        logP_list = []
        SA_list = []
        cycle_list = []
        score_list = []
        for each_smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(each_smi)
            except:
                mol_list.append(None)
                logP_list.append(None)
                cycle_list.append(None)
                continue

            if mol is None:
                mol_list.append(None)
                logP_list.append(None)
                cycle_list.append(None)
                continue
            else:
                mol_list.append(mol)
                logP_list.append((log_p(mol) - logP_mean) / logP_std)
                cycle_list.append((cycle_score(mol) - cycle_mean) / cycle_std)
        SA_list = synthetic_accessibility_batch(mol_list)

        for each_idx in range(len(mol_list)):
            if mol_list[each_idx] is not None:
                score_list.append(logP_list[each_idx] - cycle_list[each_idx] - (SA_list[each_idx] - SA_mean) / SA_std)
            else:
                score_list.append(None)
        return score_list
