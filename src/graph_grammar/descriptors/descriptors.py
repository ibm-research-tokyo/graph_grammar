#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "July 18 2018"

import gzip
import math
import networkx as nx
import os
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit.six import iteritems
from .sascorer import synthetic_accessibility, synthetic_accessibility_batch

def log_p(mol):
    ''' Calculate log p of the input mol

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    float : log p
    '''
    return Descriptors.MolLogP(mol)

def load_fscores():
    with gzip.open(os.path.join(os.path.dirname(__file__), 'fpscores.pkl.gz'), 'rb') as f:
        _fscores = pickle.load(f)
    return _fscores


def cycle_score(mol):
    ''' calculate cycle score

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    int : cycle score
    '''
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))

    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(each_cycle) for each_cycle in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length
