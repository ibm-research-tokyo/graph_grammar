# Codes for experiments
This directory is dedicated to experiments using MHG-VAE.

## Install
Just install `graph_grammar` and its dependencies by running the followings in the parent directory:

```bash
conda install scipy==1.2.1 pandas==0.23.4 numpy==1.16.2 scikit-learn
conda install pytorch==1.1.0
conda install -c rdkit rdkit==2018.03.4.0
python setup.py install
```

For local execution, see `Dockerfile` for the dependencies.
Recommended to use nvidia-docker to run experiments, because we sometimes observed different experimental results for different CUDA versrions. 

```bash
docker build -t molgrammar . --build-arg GITHUB_USER=[YOUR GITHUB USER ID] --build-arg GITHUB_TOKEN=[YOUR GITHUB TOKEN]
docker run --gpus all -v $PWD:/home/docker/tasks -t molgrammar python main.py CheckReconstructionRate --working-dir /home/docker/tasks/[WORKING DIR NAME] --use-gpu]
```

## How to run experiments

All of the experiments are managed by a workflow management tool `luigi` and its wrapper `luigine`.
In general, to run a task, the following command is used:

```bash
python main.py [task name] --working-dir [path to working directory] --workers [num workers]
```

Then, `luigi` builds a workflow to execute the task by resolving dependencies.
The results of the task and its dependent tasks are stored under `[working_dir]/OUTPUT/[each task's subdirectory]`.
These results are reused whenever possible; if `luigi` could find the results obtained using the same task parameter.
If a user wishes not to reuse the results, please remove `[working_dir]/OUTPUT`.
An execution log file is stored in `[working_dir]/ENGLOG/engine.log`.

An example of our experimental result is located in `template_working_dir`.
A user may reuse some of the results to reduce its computation time (for example, an inferred graph grammar `template_working_dir/OUTPUT/data_prep`).


### Sampling from Prior
This experiment is to evaluate the success rate of decoding random latent vectors.
Suppose latent vector $\mathbf{z}$ is sampled from the prior distribution $\mathcal{N}(0, I)$.
The decode succeeds if $\mathsf{Dec}(\mathbf{z})$ is a valid molecule.

After executing the following command, the experimental result appears under `template_working_dir/OUTPUT/sample`.
If your machine does not have GPU, then execute the following __without__ `--use-gpu` option.

```bash
python main.py SampleWithPred --working-dir template_working_dir --use-gpu
```

### Reconstruction Rate
This experiment is to estimate the reconstruction rate of MHG-VAE, i.e., how many molecules in the test set can be reconstructed using MHG-VAE.

After executing the following command, the experimental result appears under `template_working_dir/OUTPUT/reconstruction_rate`.
If your machine does not have GPU, then execute the following __without__ `--use-gpu` option.

```bash
python main.py CheckReconstructionRateWithPred --working-dir template_working_dir --use-gpu
```


### Global Molecular Optimization
This experiment executes the global molecular optimization.
In default, we run five-step Bayesian optimization ten times, and each run can be executed in a parallel manner.
Since our machine has eight cores, we executed this experiment using the following command, where we have four workers (specified as `--workers 4`), and each of the workers can use two threads (specified as `OMP_NUM_THREADS=2`).
After executing the following command, the experimental result appears under `template_working_dir/OUTPUT/multiple_bayesian_optimization`.

```bash
OMP_NUM_THREADS=2 python main.py MultipleBayesianOptimizationWithPred --working-dir template_working_dir --use-gpu --workers 4
```


### Local Molecular Optimization 
This experiment executes the local molecular optimization.
After executing the following command, the experimental result appears under `template_working_dir/OUTPUT/summary_constrained_molopt`, and `template_working_dir/ENGLOG/engine.log`.
When running the experiment, please set `ComputeTargetValues_params['target']` to be `'logP - SA'`.

```bash
python main.py SummaryConstrainedMolOpt --working-dir template_working_dir --use-gpu
```


## Experimental settings
The experimental settings can be configured by `param.py` in `[working_dir]/INPUT`.
Please refer to `template_working_dir/INPUT/param.py` for the details.

(c) Copyright IBM Corp. 2019
