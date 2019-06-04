# JaxEnt
A [JAX](https://github.com/google/jax)-based python package for maximum entropy modelling of binary data. 

## What is JaxEnt?
JaxEnt is a small, lightweight python package for fitting, sampling and constructing maximum entropy distributions with arbitrary constraints (almost; see below). As the name suggests, JaxEnt uses [JAX](https://github.com/google/jax) to get JIT compilation to CPU/GPU/TPU. JaxEnt is a research project under active development. Expect `NotImplementedError`s, and possibly some API breaking changes as JaxEnt gradually support more usecases.
 
## Installation

To install JaxEnt with a CPU version of JAX, you can use pip:

```
pip install jaxent
```

To use JaxEnt on the GPU, you will need to first [install](https://github.com/google/jax#installation) `jax` and `jaxlib` with CUDA support.

You can also install JaxEnt from source:

```
git clone https://github.com/adamhaber/jaxent.git
# install jax/jaxlib first for CUDA support
pip install -e .[dev]
```

## Examples
Maximum entropy distributions over binary variables are very common in a wide variety of fields and applications. Examples include: **(add links)**
 - Pairwise, K-Synchrony and Random Projection models in neuroscience
 - Exponential Random Graph Models (ERGMs) in networks science
 - Ising Model in statistical physics
 - Restricted Boltzmann Machines in machine learning
 
Here's an example of generating fake data from one Ising model, and fitting a different model to the same data: 

```
bla bla
```

 ## Future Work
 
In the near term, we plan to work on the following. Please open new issues for feature requests and enhancements:

 - ...
 - ...
 
 ## Thanks
 -  Ori Maoz who wrote the original, excellent matlab maxent toolbox. I plagarised large parts of his API, readme, etc.
 -  yoni and omri
 
 # put in a different file:
 ## Mathy background
Loosely speaking, maximum entropy models give the "most uniform" probabilistic models of the states or configurations of a systems, given the mean values of some set of observed functions. More precisely, it is the solution of the following constrained optimization problem:
minDkl(p,unif) subject to this and that constraints
Conceptually, they can be thought of as models that reproduce some observed behavior, but assume nothing else. This may sounds isoteric, but in the real values case, many distributions are maxent distributions - gaussians, log-normal, poisson, ...

## How does it work?
In a typical usecase, we start from an assumed parametric form (that is, a set of functions in the exponent) and desired expectation values of the model. Our goal is to find the parameters \lambda_i that will produce the desired EVs. To do so, we:  
1. Start from a random set of parameters.
2. Compute expectation values wrt these parameters.
3. Update the parameters in the direction of the gradient (which is the difference of model and desired EVs)
4. Measure if we've converged, if not - repeat.
Since the size of our sample space (and therefor the size of explicit probability vector) is 2^N, there's a "representational watershed" around N~25 - we simply can't store the whole vector in memory, which makes analytic computations of model marginals impossible. Instead, we use 

## Gory details
Various technical details - only for the highly determined.
 -  How convergence is determined.
 -  Metropolis Hastings sampling from the model.


 
