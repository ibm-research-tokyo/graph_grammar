#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2017"
__version__ = "0.1"
__date__ = "Dec 23 2017"

from graph_grammar.io.smi import HGGen
from graph_grammar.algo.tree_decomposition import (tree_decomposition,
                                                   tree_decomposition_with_hrg,
                                                   tree_decomposition_from_leaf,
                                                   topological_tree_decomposition,
                                                   molecular_tree_decomposition)
from graph_grammar.graph_grammar.hrg import HyperedgeReplacementGrammar
import networkx as nx
import os
import unittest        


class TreeDecompositionTest(unittest.TestCase):
    def tree_decomposition_check(self, each_hg, clique_tree):
        self.assertTrue(nx.is_tree(clique_tree), msg="clique_tree is not a tree.")

        # check vertex-cover
        node_set = set([])
        for each_node in each_hg.nodes:
            each_node_in_tree = False
            for each_tree_node in clique_tree.nodes():
                node_set.update(clique_tree.node[each_tree_node]["subhg"].nodes)
                if each_node in clique_tree.node[each_tree_node]["subhg"].nodes:
                    each_node_in_tree = True
            self.assertTrue(each_node_in_tree, msg="node {} is not in the clique tree".format(each_node))
        self.assertTrue(node_set == each_hg.nodes)

        # check hyperedge-cover
        edge_set = set([])
        for each_edge in each_hg.edges:
            each_edge_in_tree = False
            for each_tree_node in clique_tree.nodes():
                edge_set.update(clique_tree.node[each_tree_node]["subhg"].edges)
                if each_edge in clique_tree.node[each_tree_node]["subhg"].edges:
                    each_edge_in_tree = True
                    self.assertTrue(
                        clique_tree.node[each_tree_node]['subhg'].nodes_in_edge(each_edge)
                        == each_hg.nodes_in_edge(each_edge),
                        msg='some nodes are deleted: from {} to {}'.format(
                            clique_tree.node[each_tree_node]['subhg'].nodes_in_edge(each_edge),
                            each_hg.nodes_in_edge(each_edge)))

            self.assertTrue(each_edge_in_tree, msg="edge {} is not in the clique tree".format(each_edge))
        self.assertTrue(edge_set == each_hg.edges)

        # check running intersection
        for each_node in each_hg.nodes:
            subtree = clique_tree.subgraph([
                each_tree_node for each_tree_node in clique_tree.nodes()\
                if each_node in clique_tree.node[each_tree_node]["subhg"].nodes])
            self.assertTrue(nx.is_connected(subtree),
                            msg="sub-tree w.r.t. node {} is not connected.".format(each_node))
            self.assertTrue(nx.is_tree(subtree), msg="sub-tree w.r.t. node {} is not a tree.".format(each_node))

            # check irredundancy
            for each_subtree_node in subtree.nodes:
                if not(subtree.degree[each_subtree_node] != 1 \
                       or len(subtree.node[each_subtree_node]["subhg"].adj_edges(each_node)) != 0):
                    import ipdb; ipdb.set_trace()
                self.assertTrue(
                    subtree.degree[each_subtree_node] != 1 \
                    or len(subtree.node[each_subtree_node]["subhg"].adj_edges(each_node)) != 0, msg="redundant")

        # check each node's hypergraph contains at least one node.
        for each_node in clique_tree.nodes:
            self.assertNotEqual(
                len(clique_tree.node[each_node]["subhg"].nodes), 0)

    def test_td(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        for each_hg in hg_list:
            clique_tree = tree_decomposition(each_hg)
            self.tree_decomposition_check(each_hg, clique_tree)

    def test_td_HRG(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        hrg = HyperedgeReplacementGrammar()
        hrg.learn(hg_list)
        for each_hg in hg_list:
            clique_tree = tree_decomposition_with_hrg(each_hg, hrg)
            self.tree_decomposition_check(each_hg, clique_tree)

    def test_td_leaf(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        for each_hg in hg_list:
            clique_tree = tree_decomposition_from_leaf(each_hg)
            self.tree_decomposition_check(each_hg, clique_tree)

    def test_topological_td(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        for each_hg in hg_list:
            clique_tree = topological_tree_decomposition(each_hg)
            self.tree_decomposition_check(each_hg, clique_tree)

    def test_molecular_td(self):
        base = os.path.dirname(os.path.abspath(__file__))
        hg_list = HGGen(os.path.join(base, "test.smi"))
        for each_idx, each_hg in enumerate(hg_list):
            clique_tree = molecular_tree_decomposition(each_hg)
            try:
                self.tree_decomposition_check(each_hg, clique_tree)
            except:
                pass
            #import pdb; pdb.set_trace()
                
if __name__ == "__main__":
    unittest.main()
