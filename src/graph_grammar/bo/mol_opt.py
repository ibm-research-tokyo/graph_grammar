#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimize a molecule using Bayesian optimization
"""

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "July 18 2018"

import GPyOpt
import numpy as np
import scipy as sp
import torch
from copy import deepcopy
from GPy.kern import Matern52, RBF
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC
from rdkit import Chem
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, KernelPCA
from .preprocess import GaussianRandomProjectionWithInverse
#from .sparse_gp import SparseGP
from ..io.smi import hg_to_mol


dim_reduction_catalog = {
    'PCA': PCA,
    'KernelPCA': KernelPCA,
    'GaussianRandomProjection': GaussianRandomProjectionWithInverse
}

kern_catalog = {
    'Matern52': Matern52,
    'RBF': RBF
}

gp_catalog = {
    'GPModel': GPModel,
    'GPModel_MCMC': GPModel_MCMC
}


class MolecularOptimization(object):
    ''' Optimize molecular structure

    Attributes
    ----------
    target_func : func
        target to be minimized
    autoencoder : nn.Module
        autoencoder that has `sample` method
    num_inducing_points : int
        the number of inducing points for GP
    '''
    def __init__(self, target_func, autoencoder,
                 num_inducing_points=500,
                 seed=123):
        self.target_func = target_func
        self.autoencoder = autoencoder
        self.seed = seed
        #np.random.seed(self.seed)

    def run(self, X_train, y_train, X_test=None, y_test=None, logger=print, method='SparseGP', **kwargs):
        if method == 'SparseGP':
            raise NotImplementedError('This method is deprecated because its license is unclear.')
        #return self._run_sparse_gp(X_train, y_train, X_test, y_test, logger=logger, **kwargs)
        elif method == 'GPyOpt':
            return self._run_gpyopt(X_train, y_train, X_test, y_test, logger=logger, **kwargs)
        else:
            raise ValueError(f'{method} is not implemented. choose from "SparseGP" or "GPyOpt".')

    """
    def _run_sparse_gp(self, X_train, y_train, X_test=None, y_test=None, bo_num_iter=5, num_proposals=50,
                       gp_num_iter=100, lr=1e-3, minibatch_size=5000, num_inducing_points=500,
                       grid_size=10000, num_ei_samples=1, deterministic=True,
                       min_coef=0.8, max_coef=0.8, pca_dim=40,
                       logging_freq=10, logger=print):
        ''' run molecular optimization

        Parameters
        ----------
        X_train : array-like, shape (num_train, dim)
            feature vectors of molecules, computed by autoencoder.
        y_train : array-like, shape (num_train,)
            target values of molecules
        X_test : array-like, shape (num_test, dim)
        y_test : array-like, shape (num_test,)
            used to compute test RMSE and test log-likelihood for validation
        bo_num_iter : int
            the number of BO iterations

        Returns
        -------
        history : list of dicts
            in each dict,
            - 'mol_list' is a list of molecules that BO chooses
            - 'feature' is an array containing latent representations of the molecules
            - 'score_list' is a list of scores obtained by applying `target_func` to the molecules
        '''
        if X_test is None:
            X_test = X_train[0:1, :]
            y_test = y_train[0:1]
            valid_test = False
        else:
            valid_test = True

        if X_train.shape[0] != len(y_train.ravel()):
            raise ValueError('X_train and y_train have inconsistent shapes')
        if X_test.shape[0] != len(y_test.ravel()):
            raise ValueError('X_test and y_test have inconsistent shapes')

        #preprocess = GaussianRandomProjection(n_components=20)
        preprocess = PCA(n_components=pca_dim, random_state=np.random.randint(2**10))
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        iter_idx = 0
        history = []
        while iter_idx < bo_num_iter:
            X_train_low = preprocess.fit_transform(X_train)
            X_test_low = preprocess.transform(X_test)
            # We fit the GP
            np.random.seed(iter_idx * self.seed)
            sgp = SparseGP(X_train_low, 0 * X_train_low, y_train, num_inducing_points)#, seed=np.random.randint(2**10))
            train_err, train_ll, test_err, test_ll \
                = sgp.train_via_ADAM(X_train_low,
                                     0 * X_train_low,
                                     y_train,
                                     X_test_low,
                                     X_test_low * 0,
                                     y_test,
                                     minibatch_size=minibatch_size,
                                     max_iterations=gp_num_iter,
                                     learning_rate=lr,
                                     logging_freq=logging_freq,
                                     logger=logger)

            next_inputs_low = sgp.batched_greedy_ei(num_proposals,
                                                    min_coef * np.min(X_train_low, 0),
                                                    max_coef * np.max(X_train_low, 0),
                                                    grid_size=grid_size,
                                                    n_samples=num_ei_samples,
                                                    logger=logger)
            next_inputs = preprocess.inverse_transform(next_inputs_low)
            valid_mol_list = []
            new_feature_array = []
            for i in range(num_proposals):
                self.autoencoder.init_hidden()
                batch_z = torch.zeros(self.autoencoder.batch_size, self.autoencoder.latent_dim)
                batch_z[0, :] = torch.FloatTensor(next_inputs[i].flatten())
                _, hg_list = self.autoencoder.decode(z=batch_z,
                                                     deterministic=deterministic,
                                                     return_hg_list=True)                
                mol = hg_to_mol(hg_list[0])
                try:
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                except:
                    mol = None
                if mol is not None: 
                    valid_mol_list.append(mol)
                    new_feature_array.append(next_inputs[i])
                if self.autoencoder.use_gpu:
                    torch.cuda.empty_cache()

            new_feature_array = np.vstack(new_feature_array)
            logger(f"{len(valid_mol_list)} molecules are found")
            
            score_list = []
            for each_mol in valid_mol_list:
                score = self.target_func(each_mol)
                score_list.append(score) #target is always minused

            for each_idx, each_mol in enumerate(valid_mol_list):
                logger(f'{score_list[each_idx]}\t{Chem.MolToSmiles(each_mol)}')

            if len(new_feature_array) > 0:
                X_train = np.concatenate([X_train, new_feature_array], 0)
                y_train = np.concatenate([y_train, np.array(score_list)[:, None]], 0)

            history.append({'mol_list': valid_mol_list, 'feature': new_feature_array, 'score_list': score_list,
                            'train_err': train_err,
                            'train_ll': train_ll,
                            'test_err': test_err if valid_test else np.nan,
                            'test_ll': test_ll if valid_test else np.nan})
            iter_idx += 1
        return history
    """

    def _run_gpyopt(self, X_train, y_train, X_test=None, y_test=None, bo_num_iter=5,
                    deterministic=True,
                    min_coef=0.8, max_coef=0.8,
                    num_train=1000,
                    dim_reduction_method='KernelPCA',
                    dim_reduction_params={'kernel': 'rbf', 'n_components': 20, 'fit_inverse_transform': True},
                    fix_dim_reduction=False,
                    gp_method='GPModel',
                    gp_params={'kernel': 'Matern52', 'sparse': True, 'num_inducing': 500},
                    bo_params={'batch_size': 50},
                    logger=print):
        ''' run molecular optimization

        Parameters
        ----------
        X_train : array-like, shape (num_train, dim)
            feature vectors of molecules, computed by autoencoder.
        y_train : array-like, shape (num_train,)
            target values of molecules
        X_test : array-like, shape (num_test, dim)
        y_test : array-like, shape (num_test,)
            used to compute test RMSE and test log-likelihood for validation
        bo_num_iter : int
            the number of BO iterations

        Returns
        -------
        history : list of dicts
            in each dict,
            - 'mol_list' is a list of molecules that BO chooses
            - 'feature' is an array containing latent representations of the molecules
            - 'score_list' is a list of scores obtained by applying `target_func` to the molecules
        '''
        if X_test is None:
            X_test = X_train[0:1, :]
            y_test = y_train[0:1]
            valid_test = False
        else:
            valid_test = True

        if X_train.shape[0] != len(y_train.ravel()):
            raise ValueError('X_train and y_train have inconsistent shapes')
        if X_test.shape[0] != len(y_test.ravel()):
            raise ValueError('X_test and y_test have inconsistent shapes')
        history = []
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        dim_reduction = dim_reduction_catalog[dim_reduction_method](**dim_reduction_params)
        gp_params_ = deepcopy(dict(gp_params))
        if 'kernel' in gp_params_:
            kernel_name = gp_params_.pop('kernel')
            kernel_kwargs = gp_params_.pop('kernel_kwargs')
            gp_kernel = kern_catalog[kernel_name](input_dim=dim_reduction_params['n_components'],
                                                  **kernel_kwargs)
        model = gp_catalog[gp_method](kernel=gp_kernel, **gp_params_)

        # set up a training set
        perm = np.random.permutation(len(y_train))[:num_train]
        X_train_excerpt = X_train[perm]
        y_train_excerpt = y_train[perm]
        dim_reduction.fit(X_train_excerpt)

        for _ in range(bo_num_iter):
            if not fix_dim_reduction:
                dim_reduction.fit(X_train_excerpt)
            X_train_excerpt_low = dim_reduction.transform(X_train_excerpt)
            X_test_low = dim_reduction.transform(X_test)
            space = [{'name': f'x{each_idx}',
                      'type': 'continuous',
                      'domain': (min_coef * np.min(X_train_excerpt_low[:, each_idx], 0),
                                 max_coef * np.max(X_train_excerpt_low[:, each_idx], 0))} for each_idx in range(X_train_excerpt_low.shape[1])]

            model.updateModel(X_train_excerpt_low, y_train_excerpt, None, None)
            logger(f'model updated')
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, model=model,
                                                          domain=space, X=X_train_excerpt_low, Y=y_train_excerpt, **bo_params)

            # check ll and rmse on test set
            test_pred, test_std = model.predict(X_test_low, False)
            test_err = np.sqrt(np.mean((test_pred - y_test)**2))
            test_ll = np.mean(sp.stats.norm.logpdf(test_pred - y_test,
                                                   scale=test_std))

            # check ll and rmse on training set
            train_pred, train_std = model.predict(X_train_excerpt_low, False)
            train_err = np.sqrt(np.mean((train_pred - y_train_excerpt)**2))
            train_ll = np.mean(sp.stats.norm.logpdf(train_pred - y_train_excerpt,
                                                    scale=train_std))
            logger(f'train_err: {train_err}\t train_ll: {train_ll}')
            if valid_test:
                logger(f'test_err: {test_err}\t test_ll: {test_ll}')

            X_next_low = bo_step.suggest_next_locations(ignored_X=X_train_excerpt_low)
            X_next = dim_reduction.inverse_transform(X_next_low)

            valid_mol_list = []
            new_feature_array = []
            for i in range(len(X_next)):
                self.autoencoder.init_hidden()
                batch_z = torch.zeros(self.autoencoder.batch_size, self.autoencoder.latent_dim)
                batch_z[0, :] = torch.FloatTensor(X_next[i].flatten())
                _, hg_list = self.autoencoder.decode(z=batch_z,
                                                     deterministic=deterministic,
                                                     return_hg_list=True)
                try:
                    mol = hg_to_mol(hg_list[0])
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                except:
                    mol = None
                if mol is not None:
                    valid_mol_list.append(mol)
                    new_feature_array.append(X_next[i])
                if mol is None:
                    print('None!')
                if self.autoencoder.use_gpu:
                    torch.cuda.empty_cache()

            if new_feature_array != []:
                new_feature_array = np.vstack(new_feature_array)
            logger(f"{len(valid_mol_list)} molecules are found")
            
            score_list = []
            for each_mol in valid_mol_list:
                score = self.target_func(each_mol)
                score_list.append(score) #target is always minused

            for each_idx, each_mol in enumerate(valid_mol_list):
                logger(f'{score_list[each_idx]}\t{Chem.MolToSmiles(each_mol)}')

            if len(new_feature_array) > 0:
                X_train_excerpt = np.concatenate([X_train_excerpt, new_feature_array], 0)
                y_train_excerpt = np.concatenate([y_train_excerpt, np.array(score_list)[:, None]], 0)
                #X_train = np.concatenate([X_train, new_feature_array], 0)
                #y_train = np.concatenate([y_train, np.array(score_list)[:, None]], 0)

            history.append({'mol_list': valid_mol_list, 'feature': new_feature_array, 'score_list': score_list,
                            'train_err': train_err, 'train_ll': train_ll,
                            'test_err': test_err if valid_test else np.nan,
                            'test_ll': test_ll if valid_test else np.nan})
        return history
