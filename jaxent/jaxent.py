import jax.numpy as np
import jax.random as random
from jax import jit,vmap,grad
from jax.ops import index_update
from jax.lax import fori_loop, cond, while_loop
from jax.experimental import optimizers
from jax.scipy.special import logsumexp
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as onp
from functools import partial
import itertools as it
from tqdm import tqdm
from scipy.stats import beta

import matplotlib.pyplot as plt
from time import time

# TODO:
# [] take functions such as wl out of Model?
# [] Add sparse matrices for create_words
# [] Analytical Indep model
# [] Consider splitting Model into Model and ExpDistribution\
# [] Add ERGM example
# [v] Add KIsing example
# [] Add RBM example
# [] Profiling using asv: https://github.com/airspeed-velocity/asv
# [] Add some max_iterations flag for training
# [] implement wang landau
# [] implement memory in train_sample
# [] Add more efficient implementation of isingNN 
# [] Raise execption if an exhuastive function is called with N>20. Decorator? 

# temporary hack until installation issues on cluster are fixed
import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/RH7U2/general/cuda/10.0/"

def clopper_pearson(k,n,alpha):
    """Confidence intervals for a binomial distribution of k expected successes on n trials:
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Parameters
    ----------
    k : array_like
        number of successes
    n : array_like
        number of trials
    alpha : float
        confidence level

    Returns
    -------
    hi, lo : array_like
        lower and upper bounds on the expected number of successes
    """
    lo = beta.ppf(alpha/2, k, n-k+1)
    lo[onp.isnan(lo)] = 0   # hack to remove NaNs where we only have 0 samples
    hi = beta.ppf(1 - alpha/2, k+1, n-k)
    hi[onp.isnan(hi)] = 1   # hack to remove NaNs where the marginal is 1
    return lo, hi

class Model:
    def __init__(self,N,funcs,N_exhuastive_max=20):
        self.N = N
        self.funcs = funcs
        self.entropy = None
        self.Z = None
        self.factors = np.zeros(len(funcs))
        self.model_marginals = None
        self.empirical_marginals = None
        self.empirical_std = None
        self.p_model = None
        self.words = None
        self.trained = False
        self.converged = False
        self.training_steps = 0
        self.N_exhuastive_max = N_exhuastive_max

        self.calc_e = jit(self.calc_e)
        self._sample = jit(self._sample, static_argnums=(1,))
        self.calc_logp_unnormed = jit(self.calc_logp_unnormed)
        self.calc_logZ = jit(self.calc_logZ)
        self.calc_logp = jit(self.calc_logp)
        self._calc_p = jit(self._calc_p)
        self._calc_entropy = jit(self._calc_entropy)
        self.calc_marginals = jit(self.calc_marginals)#,static_argnums=(1,))
        self.calc_marginals_ex = jit(self.calc_marginals_ex)#,static_argnums=(1,))
        self.calc_deviations = jit(self.calc_deviations)
  
    def calc_e(self,factors,word):
        """calc the energy of a single binary word
        
        Parameters
        ----------
        factors : array_like
            factors of the distribution
        word : array_like, binary
            a single binary vector
        
        Returns
        -------
        float
            energy value
        """
        return np.sum(factors*np.array([func(word) for func in self.funcs]))

    def sample(self,key,n_samps):
        """generate samples from a distributions with the model factors
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            jax random number generator 
        n_samps : int
            number of samples to generate
        
        Returns
        -------
        array_like
            samples from the model
        """
        return self._sample(key,n_samps,self.factors)

    def _sample(self,key,n_samps,factors):
        """generate samples from a distributions with a given set of factors
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            jax random number generator 
        n_samps : int
            number of samples to generate
        factors : array_like
            factors of the distribution
        
        Returns
        -------
        array_like
            samples from the model
        """
        state = random.randint(key,minval=0,maxval=2, shape=(self.N,))
        unifs = random.uniform(key, shape=(n_samps*self.N,))
        @jit
        def run_mh(j, loop_carry):
            state, all_states = loop_carry
            all_states = index_update(all_states,j//self.N,state)  # a bit wasteful
            state_flipped = index_update(state,j%self.N,1-state[j%self.N])
            dE = self.calc_e(factors,state_flipped)-self.calc_e(factors,state)
            accept = ((dE < 0) | (unifs[j] < np.exp(-dE)))
            state = np.where(accept, state_flipped, state)
            return state, all_states

        all_states = np.zeros((n_samps,self.N))
        all_states = fori_loop(0, n_samps * self.N, run_mh, (state, all_states))
        return all_states[1]

    def calc_logp_unnormed(self,factors):
        """calc the log of the normalized probability distribution with the given factors, 
        before normalization (substraction of log of the partition function).
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.

        Parameters
        ----------
        factors : array_like
            factors of the distribution
        
        Returns
        -------
        logp_unnormed
            array of log probabilities, before substraction of log of the partition function
        """
        # e_per_word=jit(partial(self.calc_e,factors))
        # e_all_words=jit(vmap(e_per_word))(self.words)
        # logp_unnormed = -e_all_words
        logp_unnormed = -self.words@factors
        return logp_unnormed

    # def wang_landau(self,factors):
    #     raise NotImplementedError

    def calc_logZ(self,logp_unnormed):
        """calc partition function of an unnormalized probability distribution.
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.
        For N>20, use wang_landau to estimate the parition function, instead.
        
        Parameters
        ----------
        logp_unnormed : array_like
             the unnormalized probability distribution
        
        Returns
        -------
        float
            value of the partition function
        """
        logZ = logsumexp(logp_unnormed)
        return logZ
    
    def calc_logp(self,factors):
        """calc the log of the normalized probability distribution with the given factors, 
        after normalization (substraction of log of the partition function).
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.
        
        Parameters
        ----------
        factors : array_like
            factors of the distribution
        
        Returns
        -------
        array_like
            array of log probabilities, after substraction of log of the partition function
        """
        logp_unnormed = self.calc_logp_unnormed(factors)
        logZ = self.calc_logZ(logp_unnormed)
        logp = logp_unnormed - logZ
        return logp

    def _calc_p(self,factors):
        """calc the normalized probability distribution with the given factors.
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.
        
        Parameters
        ----------
        factors : array_like
            factors of the distribution
        
        Returns
        -------
        array_like
            array of probabilities.
        """
        logp = self.calc_logp(factors)
        return np.exp(logp)
    
    def calc_p(self,factors):
        """calc the normalized probability distribution with the given factors, and sets the corresponding class attribute.  
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.
        
        Parameters
        ----------
        factors : array_like
            factors of the distribution
        
        Returns
        -------
        array_like
            array of probabilities.
        """
        if self.p_model is None:
            self.p_model = self._calc_p(factors)
        return self.p_model

    def calc_marginals(self,words):
        """calc the mean parameters (expectation values of the constraint function).
        
        Parameters
        ----------
        words : array_like
            sample of binary words from which to compute the 

        Returns
        -------
        array_like
            mean parameters
        """
        marg = np.array([vmap(f)(words).mean() for f in self.funcs])
        return marg

    def calc_deviations(self,model_marg):
        """calc how many (empirical) standard deviations is the model marginal from the empirical marginal, for all marginals
        
        Parameters
        ----------
        model_marg : array_like
            model margianls
        """
        devs = np.abs(model_marg - self.empirical_marginals) / self.empirical_std
        return devs
        
    def calc_marginals_ex(self,ps):
        """calc the mean parameters (expectation values of the constraint function) analytically.
        Only possible for N<=20, since the entire probability distribution needs to fit in memory.
        
        Parameters
        ----------
        ps : array_type
            probability distribution w.r.t which the expectation values are computed
        """
        return ps@self.words
        # return np.stack([vmap(f)(self.words) for f in self.funcs])@ps

    def create_words(self):
        """create an array of all binary words, needed for exhuastive computations.
        Only possible for N<=20, since the entire sample space needs to fit in memory.
        """
        self.words = np.array(onp.fliplr(list(it.product([0,1],repeat=self.N))))
        self.words = np.stack([vmap(f)(self.words) for f in self.funcs]).T

    def calc_empirical_marginals_and_stds(self,data,data_kind,data_n_samp,alpha):
        """compute expectation values and corresponding confidence intervals from empirical observations.
        
        Parameters
        ----------
        data : array_like
            either an array of binary samples, or an array of desired marginals
        data_kind : str
            "samples" - data samples are passed
            "marginals" - desired marginals are passed
        data_n_samp : int
            number of trials, needed to compute confidence intervals
        alpha : float
            confidence level
        """
        if data_n_samp is None:
            data_n_samp = data.shape[0]

        if data_kind=="samples":
            self.empirical_marginals = self.calc_marginals(data)
        elif data_kind=="marginals":
            self.empirical_marginals = data
        (lower, upper) = clopper_pearson(self.empirical_marginals * data_n_samp, data_n_samp, alpha)
        self.empirical_std = upper - lower
            
    def train(self,data,data_kind="samples",data_n_samp=None,alpha=0.32,lr=1e-1,threshold=1.,kind=None,n_samps=5000):
        """fit a maximum entropy model to data.
        
        Parameters
        ----------
        data : array_like
            either an array of binary samples, or an array of desired marginals
        data_kind : str, optional
            "samples" - data samples are passed
            "marginals" - desired marginals are passed
        data_n_samp : int, optional
            number of trials, needed to compute confidence intervals
        alpha : float, optional
            confidence level
        lr : float, optional
            learning rate, by default 1e-1
        threshold : float, optional
            maximum allowed difference between model marginals and empirical marginals, in empirical standard deviations units. by default 1
        kind : str, optional
            "exhuastive" means analytical computation of model marginals, "sample" means MCMC estimation, by default None and estimated from the number of units N.
        n_samps : int, optional
            number of samples to generate in each MCMC estimation of model marginals, by default 5000
        """
        @jit
        def _training_step_ex(i,opt_state):
            params = get_params(opt_state)
            model_marg = self.calc_marginals_ex(self._calc_p(params))
            g = self.empirical_marginals-model_marg
            return opt_update(i, g, opt_state),model_marg
        
        @jit
        def _training_step(i,opt_state):
            params = get_params(opt_state)
            samples = self._sample(random.PRNGKey(onp.random.randint(0,10000)),5000,params)
            model_marg = self.calc_marginals(samples)
            g = self.empirical_marginals-model_marg
            return opt_update(i, g, opt_state),model_marg
        
        @jit
        def _training_loop(loop_carry):
                    i,opt_state, params,_ = loop_carry
                    opt_state,marginals = step(i,opt_state)
                    params = get_params(opt_state)
                    return i+1,opt_state, params,marginals

        if kind is None:
            kind = onp.where(self.N>20,'sample','exhuastive')

        if kind=='exhuastive':
            step = _training_step_ex
            if self.words is None:
                self.create_words()
            self.model_marginals = self.calc_marginals_ex(self._calc_p(self.factors))
        elif kind=='sample':
            step = _training_step
            self.model_marginals = self.calc_marginals(self._sample(random.PRNGKey(onp.random.randint(0,10000)),n_samps,self.factors))

        self.calc_empirical_marginals_and_stds(data,data_kind,data_n_samp,alpha)
    
        opt_init, opt_update, get_params = optimizers.adam(lr)
        training_steps, opt_state, params,marginals = while_loop(lambda x: np.max(self.calc_deviations(x[3])) > threshold,_training_loop, (0,opt_init(self.factors), self.factors,self.model_marginals))
        
        self.factors = params
        self.training_steps = training_steps
        self.model_marginals = marginals
        self.trained = True

        if kind=='exhuastive':
            self.p_model = self._calc_p(self.factors)
            self.Z = np.exp(self.calc_logZ(self.calc_logp_unnormed(self.factors)))  
            self.entropy = self._calc_entropy()

    def _calc_entropy(self):
        if self.p_model is not None:
            return -np.sum(self.p_model*np.log2(self.p_model))
    
    def wang_landau(self, bin_width=0.01, depth=10):
        # default arguments
        nsamples = 5000
        FLATNESS_CRITEREA = 0.9

        # separate into bins covering all possible values of the energy
        min_energy = np.sum(np.where(self.factors<0,self.factors,0))
        max_energy = np.sum(np.where(self.factors>0,self.factors,0))

        energy_bins = np.arange(min_energy, max_energy+bin_width, bin_width)        

        @jit
        def wl_step(j, loop_carry):
            state, curr_bin, curr_nrg, histogram, densities, bits_to_flip, unifs = loop_carry

            bit_to_flip = bits_to_flip[j]

            state_flipped = index_update(state,bit_to_flip,1-state[bit_to_flip])
            new_nrg = self.calc_e(self.factors,state_flipped)
            new_bin = (new_nrg-min_energy)//bin_width

            dE = densities[new_bin] - densities[curr_bin]  
            accept = ((dE < 0) | (unifs[j] < np.exp(-dE)))

            state = np.where(accept,new_state,state)
            curr_bin = np.where(accept,new_bin,curr_bin)
            curr_nrg = np.where(accept,new_nrg,curr_nrg)

            densities = index_update(densities,curr_bin,update_factor)
            histogram = index_update(histogram,curr_bin,1)
            return state, curr_bin, curr_nrg, histogram, densities, bits_to_flip, unifs

        @jit
        def wl_while_until_flat(loop_carry):
            mean_hist, min_hist, state, histogram, densities,key = loop_carry
            #actual WL step
            key, _ = random.split(key)
            bits_to_flip = random.randint(key,minval=0,maxval=self.N, shape=(nsamples,))
            unifs = random.uniform(key, shape=(nsamples,))

            init_nrg = self.calc_e(self.factors,state)
            init_bin = (init_nrg-min_energy)//bin_width

            state,_,_,histogram,densities,_,_ = fori_loop(0,nsamples,wl_step,(state,init_bin,init_nrg,histogram,densities,bits_to_flip,unifs))

            tmp = np.where(densities > 0,histogram,0)
            min_hist = np.min(tmp)
            mean_hist = np.mean(tmp)
            return mean_hist, min_hist, state, histogram, densities,key

        @jit
        def wl_depth_loop(j, loop_carry):
            state, histogram, densities,update_factor, key = loop_carry
            # mean_hist, min_hist ,init_state, energy_bins, densities, histogram, update_factor, nsamples, separation = 
            # stopping_crit = lambda x: x[1] <= (x[0] * FLATNESS_CRITEREA)
            mean_hist, min_hist, state, histogram, densities,key = while_loop(lambda x: x[1] <= (x[0] * FLATNESS_CRITEREA), wl_while_until_flat,(1, 0 , state, histogram, densities,key))
            update_factor = update_factor / 2
            histogram = np.zeros_like(histogram)
            return state, histogram, densities, update_factor, key

        state, histogram, densities,update_factor, key = fori_loop(0, depth, wl_depth_loop,
                (np.zeros(self.N), np.zeros_like(energy_bins), np.zeros_like(energy_bins), 1, random.PRNGKey(0)))

        return energy_bins, histogram, densities

        # # remove the non-zero states
        # energy_bins = energy_bins[densities > 0]
        # densities = densities[densities > 0]

        # # compute the partition function:
        # # first scale the densities down to avoid numerical problems
        # densities = densities - densities.min()

        # # it is unnormalized log, un-log it and normalize to 2^N
        # densities = np.exp(densities)
        # densities = densities / densities.sum()
        # densities = densities * 2 ** self.ncells

        # # now we can estimate the partition function from the densities and energy bins
        # Z = (densities * np.exp(-energy_bins)).sum()
        # logZ = -np.log(Z)

        # self.z = logZ

        # entropy = (densities * energy_bins * np.exp(-energy_bins) / (np.log(2) * Z)).sum() + np.log2(Z)
        # self.__fields['entropy'] = entropy

class Ising(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_2 = lambda i,j,x:x[i]*x[j]

        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        marg_2s = [jit(partial(marg_2,i,j)) for i,j in list(it.combinations(range(N),r=2))]
        super().__init__(N,funcs=marg_1s+marg_2s)
        
    def calc_e(self,factors,word):
        return factors@np.concatenate([word,np.outer(word,word)[onp.triu_indices(self.N,1)]])#self.idx[0],self.idx[1]]])
    
class MERP(Model):
    def __init__(self,N,nprojections=None, indegree=5, threshold=0.1, projections=None, projections_thresholds=None,seed=0):
        if projections is not None:
            self.projections = projections
            self._n_projections = projections.shape[0]
        else:
            if nprojections is None:
                self._n_projections = N * (N + 1) // 2  # default - same number of factors as ising
            elif type(nprojections) == int:
                self._n_projections = nprojections
            else:
                raise TypeError('nprojections must be an integer')

            key = random.PRNGKey(seed)
            projections = random.normal(key,shape=(self._n_projections, N))+1
            sparsity = indegree / N
            self.projections = np.where(random.uniform(key,shape=(self._n_projections, N)) > sparsity, 0, projections)
        if projections_thresholds is None:
            self.projections_thresholds = np.ones(self._n_projections) * threshold * indegree
        else:
            self.projections_thresholds = projections_thresholds

        proj = lambda i,x:np.where(self.projections[i]@x>self.projections_thresholds[i],1,0)

        projs = [jit(partial(proj,i)) for i in range(self._n_projections)]
        super().__init__(N,funcs=projs)
        
    def calc_e(self,factors,word):
        # return np.sum(factors[:self.N]*word[:self.N]) + np.sum(factors[self.N:]*np.outer(word,word)[self.idx[0],self.idx[1]])
        return factors@np.where(self.projections@word>self.projections_thresholds,1,0)

class Indep(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        super().__init__(N,funcs=marg_1s)
        self.calc_e = jit(self.calc_e)

    def calc_e(self,factors,word):
        return factors@word    

class KIsing(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_2 = lambda i,j,x:x[i]*x[j]
        marg_3 = lambda i,x:np.sum(x)==i
        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        marg_2s = [jit(partial(marg_2,i,j)) for i,j in list(it.combinations(range(N),r=2))]
        marg_3s = [jit(partial(marg_3,i)) for i in range(N+1)]
        super().__init__(N,funcs=marg_1s+marg_2s+marg_3s)
        self.k_sync_factors_start_idx = len(marg_1s + marg_2s)
        
    def calc_e(self,factors,word):
        return factors[:self.k_sync_factors_start_idx]@np.concatenate([word,np.outer(word,word)[onp.triu_indices(self.N,1)]])+factors[self.k_sync_factors_start_idx:][np.sum(word)]#self.idx[0],self.idx[1]]])
    
class IsingNN(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_2 = lambda i,j,x:x[i]*x[j]
        pairs_h = []
        pairs_v = []
        N_new = int(N**0.5)
        for i in range(N_new):
            for j in range(N_new):
                pairs_h.append((i*N_new+j,i*N_new+(j+1)%N_new))
                pairs_v.append((i+j*N_new,(i+(j+1)*N_new)%N))
        pairs = pairs_h + pairs_v
        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        marg_2s = [jit(partial(marg_2,i,j)) for i,j in pairs]
        self.pairs = np.array(onp.array(pairs))
        super().__init__(N,funcs=marg_1s+marg_2s)
        
    def calc_e(self,factors,word):
        fields = factors[:self.N]@word
        corrs = 0
        for f,(i,j) in zip(factors[self.N:],self.pairs):
            corrs += f*word[i]*word[j]
        return fields+corrs
    
class ERGM(Model):
    def __init__(self,N,funcs):
        self.num_of_nodes = N
        self.num_of_edges = N*(N-1)
        self.diag_idx = onp.arange(0.,N*N,N,dtype=int)
        super().__init__(self.num_of_edges,funcs=funcs)

    def insert_diag(self,x):
        return onp.insert(x,0,axis=1)      ## to replace by np.insert when it's implemented

