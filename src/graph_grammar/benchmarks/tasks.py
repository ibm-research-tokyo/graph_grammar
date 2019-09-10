#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2019"
__version__ = "0.1"
__date__ = "Mar 15 2019"

from .scores import BatchStandardizedPenalizedLogP
from guacamol.assess_goal_directed_generation import _evaluate_goal_directed_benchmarks
from guacamol.goal_directed_score_contributions import uniform_specification
import os


class MyGoalDirectedBenchmarkSuite(object):

    ''' This class composes your own benchmaark suite.

    Attributes
    ----------
    benchmark_list : list of GoalDirectedBenchmark objects
    optimizer : GoalDirectedGenerator
    benchmark_suite_name : str
    logger : logger
    '''

    def __init__(self, benchmark_list, optimizer, benchmark_suite_name, logger=print):
        self.benchmark_list = benchmark_list
        self.optimizer = optimizer
        self.benchmark_suite_name = benchmark_suite_name
        self.logger = logger

    def run(self, json_path=None):
        ''' Run benchmark

        Parameters
        ----------
        json_path : str, optional
            if `json_path` is given, the results are saved to the path 
        '''
        results = _evaluate_goal_directed_benchmarks(self.optimizer, self.benchmark_list)
        if json_path is not None:
            if os.path.splitext(json_path)[1] != '.json':
                raise ValueError('json_path must have extension .json')
            from collections import OrderedDict
            import json
            from guacamol.utils.data import get_time_string

            benchmark_results = OrderedDict()
            benchmark_results['guacamol_version'] = guacamol.__version__
            benchmark_results['benchmark_suite_version'] = self.benchmark_suite_name
            benchmark_results['timestamp'] = get_time_string()
            benchmark_results['results'] = [vars(result) for result in results]
            logger(f'Save results to file {json_path}')
            with open(json_path, 'wt') as f:
                f.write(json.dumps(benchmark_results, indent=4))
        return results

def get_standardized_penalized_logp_maximization(top_k=50):
    return GoalDirectedBenchmark(name='standardized penalized logp maximization',
                                 objective=BatchStandardizedPenalizedLogP(),
                                 contribution_specification=uniform_specification(top_k))
