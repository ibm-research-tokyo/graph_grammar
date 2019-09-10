#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" setup file. """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__version__ = "0.1"
__date__ = "Dec 11, 2017"
__copyright__ = "(c) Copyright IBM Corp. 2017"

from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name = "graph_grammar",
    version = '1.0',
    author = 'Hiroshi Kajino',
    url='https://github.com/ibm-research-tokyo/graph_grammar',
    author_email='hiroshi.kajino.1989@gmail.com',
    package_dir = {"":"src"},
    packages = find_packages("src", exclude=["tests.*", "tests"]),
    test_suite = "tests",
    install_requires=_requires_from_file('requirements.txt'),
    include_package_data=True,
    license='CC BY-NC-SA 4.0',
)
