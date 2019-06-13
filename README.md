# JaxEnt
A [JAX](https://github.com/google/jax)-based python package for maximum entropy modeling of multivariate binary data. 

## What is JaxEnt?
JaxEnt is a small, lightweight python package for fitting, sampling from and computing various quantities of maximum entropy distributions with arbitrary constraints. As the name suggests, JaxEnt uses [JAX](https://github.com/google/jax) to get JIT compilation of various function to CPU/GPU/TPU.  JaxEnt implements several popular maximum entropy models (see below), and extending it to other usecases is straightforward.

JaxEnt is a research project under active development (as is JAX itself). Expect `NotImplementedError`-s, and possibly future API breaking changes as JaxEnt gradually supports more usecases. Contributions, feature requests, additions, corrections and suggestions are welcomed. 

 
## Installation

Installation is simple:
```
git clone https://github.com/adamhaber/jaxent.git
cd jaxent
pip install .
```

## Testing
To make sure everything works as planned, run: 
```
cd jaxent
pytest
```

## Examples
Maximum entropy distributions over binary variables are very common in a wide variety of fields and applications. Examples include:
 - [Pairwise](https://www.princeton.edu/~wbialek/rome/refs/schneidman+al_06.pdf), [K-Ising](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003408&type=printable) and [Random Projection](https://www.biorxiv.org/content/10.1101/478545v1.article-info) models in neuroscience
 - [Exponential Random Graph Models](https://en.wikipedia.org/wiki/Exponential_random_graph_models) (ERGMs) in networks science
 - [Ising Model](https://en.wikipedia.org/wiki/Ising_model) in statistical physics
 - [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in machine learning
 
Here's an example of generating fake data from one Ising model, and fitting a different model to the same data: 

```python
import jaxent
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
import jax

N = 15
n_data_points = 10000

# create an all-to-all-connectivity Ising model with random biases and couplings
m = jaxent.Ising(N)
m.factors = np.array(onp.concatenate([onp.random.normal(3,1,N),onp.random.normal(-0.1,0.05,N*(N-1)//2)]))

# sample from the model
emp_data = m.sample(jax.random.PRNGKey(0),n_data_points)

# create a new model and train it using the data generate from the first model
m2 = jaxent.Ising(N)
m2.train(emp_data)
```
The marginals of `m2` are all within the (normalized) errorbars of the original data:

![readme figure](https://user-images.githubusercontent.com/20320402/59023548-b794e980-8858-11e9-9350-6c7d252a25cc.png)

 ## Future Work

 - Further improve performance of sampling and training methods
 - Implememt Wang-Landau estimator of the partition function for larger models
 - Expand tests suite
 - Sparse matrices support
 - Add notebooks and examples
  
 ## Thanks
 - [Ori Maoz](https://github.com/orimaoz) who wrote the original, excellent MATLAB [maxent toolbox](https://orimaoz.github.io/maxent_toolbox/). I plagiarized large parts of his API, design choices, with permission of course.

 
