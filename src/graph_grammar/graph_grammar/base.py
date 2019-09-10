#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2017"
__version__ = "0.1"
__date__ = "Dec 11 2017"

from abc import ABCMeta, abstractmethod

class GraphGrammarBase(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def sample(self):
        pass
