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
