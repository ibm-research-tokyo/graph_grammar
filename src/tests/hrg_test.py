#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 1 2018"

from graph_grammar.io.smi import HGGen
from graph_grammar.graph_grammar.hrg import extract_prod_rule, HyperedgeReplacementGrammar, IncrementalHyperedgeReplacementGrammar
from graph_grammar.algo.tree_decomposition import tree_decomposition
from networkx.algorithms.isomorphism import GraphMatcher
import os
import unittest

class HRGTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    '''
    def test_extract_prod_rule(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        
        for each_hg in hg_list:
            clique_tree = tree_decomposition(each_hg)
            for each_node in clique_tree.nodes:
                parent = each_node
                myself = sorted(list(dict(clique_tree[parent]).keys()))[0]
                children = list(dict(clique_tree[myself]).keys())
                children.remove(parent)
                prod_rule = extract_prod_rule(
                    clique_tree.node[parent]["subhg"], clique_tree.node[myself]["subhg"],
                    [clique_tree.node[each_child]["subhg"] for each_child in children])
                self.assertTrue(check_prod_rule(prod_rule))
    '''
    def test_hrg(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        #if not os.path.exists('hg'):
        #    os.mkdir('hg')
        #for each_idx, each_hg in enumerate(hg_list):
        #    each_hg.draw(os.path.join('hg', f'{each_idx}'))
        hrg = HyperedgeReplacementGrammar()
        prod_rule_seq_list = hrg.learn(hg_list)
        print("the number of prod rules is {}".format(hrg.num_prod_rule))
        '''
        if not os.path.exists('prod_rules'):
            os.mkdir('prod_rules')
        if not os.path.exists('subhg'):
            os.mkdir('subhg')
        for each_idx, each_prod_rule in enumerate(hrg.prod_rule_corpus.prod_rule_list):
            self.assertTrue(check_prod_rule(each_prod_rule))
            each_prod_rule.draw(os.path.join('prod_rules', f'{each_idx}'))
        for each_idx, each_subhg in enumerate(hrg.clique_tree_corpus.subhg_list):
            each_subhg.draw(os.path.join('subhg', f'{each_idx}'), True)
        import gzip, pickle
        with gzip.open(os.path.join('prod_rules', 'hrg.pklz'), 'wb') as f:
            pickle.dump(hrg, f)
        '''

    def test_iso(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        hg_list = list(hg_list)
        hrg = HyperedgeReplacementGrammar()
        prod_rule_seq_list = hrg.learn(hg_list)
        not_iso = 0
        for idx, each_prod_rule_seq in enumerate(prod_rule_seq_list):
            hg = hrg.construct(each_prod_rule_seq)
            self.assertEqual(len(hg.nodes), len(list(hg_list)[idx].nodes))
            self.assertEqual(len(hg.edges), len(list(hg_list)[idx].edges))
            gm = GraphMatcher(hg.hg, list(hg_list)[idx].hg)
            try:
                isomap = next(gm.isomorphisms_iter())
            except StopIteration:
                isomap = None
            if isomap is None:
                print("not isomorphic")
                not_iso += 1
            self.assertEqual(not_iso, 0)
        print("not_iso = {}".format(not_iso))

    def test_revert(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        hrg = HyperedgeReplacementGrammar()
        prod_rule_seq_list = hrg.learn(hg_list)

        for each_hg_id in range(len(list(hg_list))):
            tmp = list(hg_list)[each_hg_id]
            each_prod_rule_id = -1
            # reverting may yield a different intermediate hypergraph, when it may matches a different subgraph.
            # but, the first step must be reverted.
            tmp, success = hrg.prod_rule_corpus.prod_rule_list[prod_rule_seq_list[each_hg_id][each_prod_rule_id]].revert(tmp)
            self.assertTrue(success, 'fails hg={}, prod_rule={}'.format(
                each_hg_id, prod_rule_seq_list[each_hg_id][each_prod_rule_id]))
            # this not necessarily works because the first revert can be applied to multiple portions.
            #each_prod_rule_id -= 1
            #tmp, success = hrg.prod_rule_corpus.prod_rule_list[prod_rule_seq_list[each_hg_id][each_prod_rule_id]].revert(tmp)
            #self.assertTrue(success, 'fails hg={}, prod_rule={}'.format(
            #    each_hg_id, prod_rule_seq_list[each_hg_id][each_prod_rule_id]))
'''
class IncrementalHRGTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hrg(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        hrg = IncrementalHyperedgeReplacementGrammar()
        prod_rule_seq_list = hrg.learn(hg_list)
        print("the number of prod rules is {}".format(len(hrg.prod_rule_list)))
        for each_prod_rule in hrg.prod_rule_list:
            self.assertTrue(check_prod_rule(each_prod_rule))

    def test_iso(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        hg_list = list(hg_list)
        hrg = IncrementalHyperedgeReplacementGrammar()
        prod_rule_seq_list = hrg.learn(hg_list)
        not_iso = 0
        for idx, each_prod_rule_seq in enumerate(prod_rule_seq_list):
            hg = hrg.construct(each_prod_rule_seq)
            self.assertEqual(len(hg.nodes), len(list(hg_list)[idx].nodes))
            self.assertEqual(len(hg.edges), len(list(hg_list)[idx].edges))
            gm = GraphMatcher(hg.hg, list(hg_list)[idx].hg)
            try:
                isomap = next(gm.isomorphisms_iter())
            except StopIteration:
                isomap = None
            if isomap is None:
                print("not isomorphic")
                not_iso += 1
            self.assertEqual(not_iso, 0)
        print("not_iso = {}".format(not_iso))
'''        
def check_prod_rule(prod_rule):
    ok_rule = True
    ext_node_list = []
    if not prod_rule.is_start_rule:
        for each_node in prod_rule.lhs.nodes:
            ext_node_list.append(each_node)
            if len(prod_rule.rhs.adj_edges(each_node)) != 1: ok_rule = False
    for each_node in prod_rule.rhs.nodes - set(ext_node_list):
        if len(prod_rule.rhs.adj_edges(each_node)) != 2: ok_rule = False
    for each_edge in prod_rule.rhs.edges:
        if "nt_idx" in prod_rule.rhs.edge_attr(each_edge) \
           and prod_rule.rhs.edge_attr(each_edge)["terminal"]:
            ok_rule = False
    return ok_rule
    
if __name__ == "__main__":
    unittest.main()
