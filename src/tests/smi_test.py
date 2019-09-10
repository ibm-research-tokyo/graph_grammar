#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 1 2018"

from graph_grammar.io.smi import HGGen, standardize_stereo
import os
import unittest

class LoadSmilesTest(unittest.TestCase):

    def setUp(self):
        self.base = os.path.dirname(os.path.abspath(__file__))
        pass

    def tearDown(self):
        pass

    def test_smi_to_hg(self):
        hg_list = HGGen(os.path.join(self.base, "test.smi"))

    def test_stereo(self):
        from rdkit import Chem
        from copy import deepcopy
        mol_gen = Chem.SmilesMolSupplier(os.path.join(self.base, "test.smi"), titleLine=False)
        for each_mol in mol_gen:
            mol_bk = deepcopy(each_mol)
            each_mol = standardize_stereo(each_mol)
            self.assertTrue(mol_bk.HasSubstructMatch(each_mol, useChirality=True))
    
if __name__ == "__main__":
    unittest.main()
