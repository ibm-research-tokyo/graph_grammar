#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 1 2018"

from graph_grammar.hypergraph import Hypergraph
import unittest

class HypergraphTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_graph_construction(self):
        hg = Hypergraph()
        hg.add_node('bond_0', attr_dict=dict(aromatic=True))
        hg.add_node('bond_1', attr_dict=dict(aromatic=False))
        hg.add_edge(['bond_0', 'bond_1'], attr_dict={'smarts' : 'c', 'aromatic': True})

if __name__ == "__main__":
    unittest.main()
