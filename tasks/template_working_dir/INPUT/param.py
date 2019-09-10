#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Jan 25 2018"

from copy import deepcopy

# ** ALL OF THE PARAMETERS in DataPreprocessimg_params SHOULD NOT BE CHANGED **
DataPreprocessing_params = {
    'kekulize': True, # Use kekulized representation or not
    'add_Hs': False, # Add hydrogens explicitly or not
    'all_single': True, # Represent every bond as a labeled single edge
    'tree_decomposition': 'molecular_tree_decomposition', # Tree decomposition algorithm
    'tree_decomposition_kwargs': {}, # Parameters for the tree decomposition algorithm
    'ignore_order': False # Ignore the orders of nodes in production rules
}

# Parameters for training a variational autoencoder
Train_params = {
    'model': 'GrammarSeq2SeqVAE', # Model name
    'model_params' : { # Parameter for the model
        'latent_dim': 72, # Dimension of the latent dimension
        'max_len': 80, # maximum length of input sequences (represented as sequences of production rules)
        'batch_size': 128, # batch size for training
        'padding_idx': -1, # integer used for padding
        'start_rule_embedding': False, # Embed the starting rule into the latent dimension explicitly
        'encoder_name': 'GRU', # Type of encoder
        'encoder_params': {'hidden_dim': 384, # hidden dim
                           'num_layers': 3, # the number of layers
                           'bidirectional': True, # use bidirectional one or not
                           'dropout': 0.1}, # dropout probability
        'decoder_name': 'GRU', # Type of decoder
        'decoder_params': {'hidden_dim': 384, # hidden dim
                           'num_layers': 3, # the number of layers
                           'dropout': 0.1}, # dropout probability
        'prod_rule_embed': ['Embedding',
                            'MolecularProdRuleEmbedding',
                            'MolecularProdRuleEmbeddingLastLayer',
                            'MolecularProdRuleEmbeddingUsingFeatures'][0], # Embedding method of a production rule. The simple embedding was the best, but each production rule could be embedded using GNN
        'prod_rule_embed_params': {'out_dim': 128, # embedding dimension
                                   'layer2layer_activation': 'relu', # not used for `Embedding`
                                   'layer2out_activation': 'softmax', # not used for `Embedding`
                                   'num_layers': 4}, # not used for `Embedding`
        'criterion_func': ['VAELoss', 'GrammarVAELoss'][1], # Loss function
        'criterion_func_kwargs': {'beta': 0.01}}, # Parameters for the loss function. `beta` specifies beta-VAE.
    'sgd': 'Adam', # SGD algorithm
    'sgd_params': {'lr': 5e-4 # learning rate of SGD
    },
    'seed_list': [141, 123, 425, 552, 1004, 50243], # seeds used for restarting training
    'inversed_input': True, # the input sequence is in the reversed order or not.
    'num_train': 220011, # the number of training examples
    'num_val': 24445, # the number of validation examples
    'num_test': 5000, # the number of test examples
    'num_early_stop': 220011, # the number of examples used to find better initializations (=seed)
    'num_epochs': 30 # the number of training epochs
}

TrainWithPred_params = deepcopy(Train_params) # used to train VAE + predictor
TrainWithPred_params['model'] = 'GrammarSeq2SeqVAEWithPred'
TrainWithPred_params['model_params']['predictor_list'] = ['Linear'] # Configuration of layers of predictors
TrainWithPred_params['model_params']['predictor_out_dim_list'] = [1] # each layer's output dimension

Sample_params = {
    'seed': 123
}

ComputeTargetValues_params = {
    'target': 'logP - SA - cycle', # specifying target value
    'num_train': Train_params['num_train']
}

CheckReconstructionRate_params = {
    'batch_size': 128,
    'seed': 123,
    'deterministic': True # is decoder deterministic or stochastic?
}

Encode_params = {
    'batch_size': 128
}

ConstructDatasetForBO_params = {
    'batch_size': 128,
    'seed': 123
}

MultipleBayesianOptimization_params = {
    'seed_list' : [123, 323, 53418, 100943, 13894, 14890, 68974, 4780, 4213, 12387460] # the length of this list determines the number of Bayesian optimization executions
}

# The following parameter is used for global optimization in the unlimited oracle case.
# It originally used the Bayesian optimization module of GrammarVAE (Kushner et al., ICML-17),
# but since its license is not specified in their repository, I had to remove the module from my repository.
# Therefore this setting is deprecated so far.
'''
BayesianOptimization_params = {
    'run_params': {'method': 'SparseGP',
                   'num_inducing_points': 500,
                   'gp_num_iter': 50,
                   'bo_num_iter': 5,
                   'lr': 1e-3,
                   'min_coef': 0.8,
                   'max_coef': 0.8,
                   'pca_dim': 24,
                   'grid_size': 10000
    }
}
'''

# The following parameter is used for global optimization in the limited oracle case.
# It uses GPyOpt
BayesianOptimization_params = {
    'run_params': {'method': 'GPyOpt',
                   'bo_num_iter': 250, # The number of Bayesian optimization iterations
                   'num_train': 250, # The number of the initial training examples
                   'dim_reduction_method': 'PCA', # a method to reduce the dimensionality of the latent space
                   #'dim_reduction_params': {'kernel': 'rbf', 'n_components': 10, 'fit_inverse_transform': True},
                   'dim_reduction_params': {'n_components': 56}, # parameters for the dimension reduction method
                   'fix_dim_reduction': False, # whether or not to re-run the dimension reduction method for each iteration
                   'gp_method': 'GPModel', # GP name
                   'gp_params': {'kernel': 'Matern52', 'kernel_kwargs':{}, 'sparse': False}, # GP parameter
                   'bo_params': {'batch_size': 1}, # BO parameter
                   'min_coef': 0.8, # the size of a search space relative to the empirical latent representations (negative sphere).
                   'max_coef': 0.8, # the size of a search space relative to the empirical latent representations (positive sphere).
                   'deterministic': True # whether the decoder is deterministic or not
    }
}

# Parameer for local optimization
ConstrainedMolOpt_params = {
    'inseq_opt_params': {'lr': 1e-1,
                         'num_iter': 80}
}


# Parameter for running GuacaMol.
# GuacaMol benchmark runs, but I couldn't finish it because of the size of its training set
GuacaMol_params = {
    'batch_size': 128,
    'bo_run_params': BayesianOptimization_params['run_params'],
    'suite': ['v1', 'v2', 'trivial'][1],
    'deterministic': True
}
