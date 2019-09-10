# Molecular Optimization Using Molecular Hypergraph Grammar
This repository contains an implementation of "Molecular Hypergraph Grammar with Its Application to Molecular Optimization", which was presented in ICML-19.

# Requirements
- numpy
- scipy
- scikit-learn
- pandas
- pytorch
- RDKit
- networkx
- luigi
- graphviz
- guacamol
- luigine
- GPy
- GPyOpt
- paramz


# Install

```bash
conda install scipy==1.2.1 pandas==0.23.4 numpy==1.16.2 scikit-learn
conda install pytorch==1.1.0
conda install -c rdkit rdkit==2018.03.4.0
python setup.py install
```

# Unit tests

```bash
python setup.py test
```

# Examples
See `tasks` directory.

# References
Hiroshi Kajino: "Molecular Hypergraph Grammar with Its Application to Molecular Optimization", ICML-19, Long Beach, CA, 2019.

# Notes
This repository is licensed under Creative Commons BY-NC-SA 4.0.
This repository contains a data set, which was acquired from ZINC database under its terms and conditions (http://wiki.bkslab.org/index.php/Terms_And_Conditions).

(c) Copyright IBM Corp. 2019
