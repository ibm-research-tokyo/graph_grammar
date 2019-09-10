#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 1 2018"

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

class GaussianRandomProjectionWithInverse(GaussianRandomProjection):
    def inverse_transform(self, X):
        proj_array = self.components_
        return (np.linalg.pinv(proj_array) @ X.T).T
