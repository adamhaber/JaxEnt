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
# [] Add sparse matrices for create_words
# [] Consider splitting Model into Model and ExpDistribution\
# [] Add ERGM example
# [] Add KIsing example
# [] Add RBM example
# [] 

# temporary hack until installation issues on cluster are fixed
import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/RH7U2/general/cuda/10.0/"

def clopper_pearson(k,n,alpha):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = beta.ppf(alpha/2, k, n-k+1)
    lo[onp.isnan(lo)] = 0   # hack to remove NaNs where we only have 0 samples
    hi = beta.ppf(1 - alpha/2, k+1, n-k)
    hi[onp.isnan(hi)] = 1   # hack to remove NaNs where the marginal is 1
    return lo, hi

class Model:

    def __init__(self,N,funcs):
        self.N = N
        self.funcs = funcs
        self.entropy = None
        self.Z = None
        self.factors = np.zeros(len(funcs))
        self.model_marg = None
        self.empirical_marginals = None
        self.empirical_std = None
        self.p_orig = None
        self.p_model = None
        self.words = None
        self.trained = False
        self.converged = False
        self.training_steps = 0

        self.calc_e = jit(self.calc_e)
        self.sample = jit(self.sample, static_argnums=(1,))
        self.calc_logp_unnormed = jit(self.calc_logp_unnormed)
        self.calc_logZ = jit(self.calc_logZ)
        self.calc_logp = jit(self.calc_logp)
        self._calc_p = jit(self._calc_p)
        self.calc_marginals = jit(self.calc_marginals)#,static_argnums=(1,))
        self.calc_normalized_errors = jit(self.calc_normalized_errors)#,static_argnums=(1,))
        self.calc_marginals_ex = jit(self.calc_marginals_ex)#,static_argnums=(1,))
        self.calc_normalized_errors_ex = jit(self.calc_normalized_errors_ex)#,static_argnums=(1,))
        self.loss_dkl = jit(self.loss_dkl)
        self.calc_deviations = jit(self.calc_deviations)
        self.loss_marg_max = jit(self.loss_marg_max)
        self.loss_marg_mean = jit(self.loss_marg_mean)
        self.loss_marg_max_ex = jit(self.loss_marg_max_ex)
        self.loss_marg_mean_ex = jit(self.loss_marg_mean_ex)


    def calc_e(self,factors,word):
        return np.sum(factors*np.array([func(word) for func in self.funcs]))

    def sample(self,key,n_samps,factors):
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
        e_per_word=jit(partial(self.calc_e,factors))
        e_all_words=jit(vmap(e_per_word))(self.words)
        logp_unnormed = -e_all_words
        return logp_unnormed

    def calc_logZ(self,logp_unnormed):
        logZ = logsumexp(logp_unnormed)
        return logZ
    
    def calc_logp(self,factors):
        logp_unnormed = self.calc_logp_unnormed(factors)
        logZ = self.calc_logZ(logp_unnormed)
        logp = logp_unnormed - logZ
        return logp

    def _calc_p(self,factors):
        logp = self.calc_logp(factors)
        return np.exp(logp)
    
    def calc_p(self,factors):
        if self.p_model is None:
            self.p_model = self._calc_p(factors)
        return self.p_model

    def calc_marginals(self,words):
        return np.array([vmap(f)(words).mean() for f in self.funcs])

    def calc_deviations(self,model_marg):
        return np.abs(model_marg - self.empirical_marginals) / self.empirical_std
    
    def calc_normalized_errors(self,factors):
        samples = self.sample(random.PRNGKey(onp.random.randint(0,10000)),10000,factors)
        model_marg = self.calc_marginals(samples)
        normalized_errors = self.calc_deviations(model_marg)
        return normalized_errors
    
    def calc_marginals_ex(self,words,ps):
        return np.stack([vmap(f)(words) for f in self.funcs])@ps

    def calc_normalized_errors_ex(self,factors):
        model_marg = self.calc_marginals_ex(self.words, self._calc_p(factors))
        normalized_errors = self.calc_deviations(model_marg)
        return normalized_errors
    
    def loss_marg_mean(self,factors):
        return np.mean(self.calc_normalized_errors(factors))

    def loss_marg_mean_ex(self,factors):
        return np.mean(self.calc_normalized_errors_ex(factors))

    def loss_marg_max(self,factors):
        return np.max(self.calc_normalized_errors(factors))

    def loss_marg_max_ex(self,factors):
        return np.max(self.calc_normalized_errors_ex(factors))

    def loss_dkl(self,factors):
        p_model = self._calc_p(factors,self.words)
        return np.nansum(p_model*np.log2(p_model/self.p_orig))

    def create_words(self):
        return None

    def train_exhuastive(self,data,data_kind="samples",data_n_samp=None,alpha=0.32,loss_kind="mean",lr=1e-1,threshold=1.):
        if data_kind=="samples":
            self.empirical_marginals = self.calc_marginals(data)
            (lower, upper) = clopper_pearson(self.empirical_marginals * data.shape[0], data.shape[0], alpha)
            self.empirical_std = upper - lower
        elif data_kind=="marginals":
            self.empirical_marginals = data
            (lower, upper) = clopper_pearson(self.empirical_marginals * data_n_samp, data_n_samp, alpha)
            self.empirical_std = upper - lower
            # self.empirical_std = data_std
        if self.words is None:
            self.create_words()

        opt_init, opt_update, get_params = optimizers.adam(lr)

        @jit
        def step(i,opt_state):
            params = get_params(opt_state)
            model_marg = self.calc_marginals_ex(self.words, self._calc_p(params))
            g = self.empirical_marginals-model_marg
            return opt_update(i, g, opt_state),model_marg

        opt_state = opt_init(self.factors)

        @jit
        def training_loop(loop_carry):
                    i,opt_state, params,_ = loop_carry
                    opt_state,marginals = step(i,opt_state)
                    params = get_params(opt_state)
                    return i+1,opt_state, params,marginals
        # training_steps, opt_state, params = while_loop(lambda x: x[0] > 0,training_loop, (0,opt_state, self.factors))
        # opt_state, params = fori_loop(0, epochs, training_loop, (opt_state, self.factors))
        self.model_marg = self.calc_marginals_ex(self.words, self._calc_p(self.factors))
        training_steps, opt_state, params,marginals = while_loop(lambda x: np.max(self.calc_deviations(x[3])) > threshold,training_loop, (0,opt_state, self.factors,self.model_marg))

        # training_steps, opt_state, params = fori_loop(0, 1000, training_loop, (0,opt_state, self.factors))

        # while np.max(self.calc_deviations(self.model_marg))>threshold:
        #     training_steps, opt_state, params = fori_loop(0, 1000, training_loop, (training_steps,opt_state,params))

        self.factors = params
        self.training_steps = training_steps
        self.p_model = self._calc_p(self.factors)
        self.model_marg = marginals
        self.Z = np.exp(self.calc_logZ(self.calc_logp_unnormed(self.factors)))
        self.entropy = -np.sum(self.p_model*np.log(self.p_model))

    def train_sample(self,data,data_kind="samples",data_n_samp=None,alpha=0.32,loss_kind="mean",lr=1e-1,threshold=1.):
        if data_kind=="samples":
            self.empirical_marginals = self.calc_marginals(data)
            (lower, upper) = clopper_pearson(self.empirical_marginals * data.shape[0], data.shape[0], alpha)
            self.empirical_std = upper - lower
        elif data_kind=="marginals":
            self.empirical_marginals = data
            (lower, upper) = clopper_pearson(self.empirical_marginals * data_n_samp, data_n_samp, alpha)
            self.empirical_std = upper - lower

        opt_init, opt_update, get_params = optimizers.adam(lr)

        @jit
        def step(i,opt_state):
            params = get_params(opt_state)
            samples = self.sample(random.PRNGKey(onp.random.randint(0,10000)),5000,params)
            model_marg = self.calc_marginals(samples)
            g = self.empirical_marginals-model_marg
            return opt_update(i, g, opt_state),model_marg

        opt_state = opt_init(self.factors)

        @jit
        def training_loop(loop_carry):
                    i,opt_state, params,_ = loop_carry
                    opt_state,marginals = step(i,opt_state)
                    params = get_params(opt_state)
                    return i+1,opt_state, params,marginals
        self.model_marg = self.calc_marginals(self.sample(random.PRNGKey(onp.random.randint(0,10000)),5000,self.factors))
        training_steps, opt_state, params,marginals = while_loop(lambda x: np.max(self.calc_deviations(x[3])) > threshold,training_loop, (0,opt_state, self.factors,self.model_marg))

        self.factors = params
        self.training_steps = training_steps
        self.model_marg = marginals
        # self.p_model = self._calc_p(self.factors)
        # self.Z = np.exp(self.calc_logZ(self.calc_logp_unnormed(self.factors)))          ##need wang landau
        # self.entropy = -np.sum(self.p_model*np.log(self.p_model))

class Ising(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_2 = lambda i,j,x:x[i]*x[j]

        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        marg_2s = [jit(partial(marg_2,i,j)) for i,j in list(it.combinations(range(N),r=2))]
        super().__init__(N,funcs=marg_1s+marg_2s)
        
    def calc_e(self,factors,word):
        # return np.sum(factors[:self.N]*word[:self.N]) + np.sum(factors[self.N:]*np.outer(word,word)[self.idx[0],self.idx[1]])
        return factors@np.concatenate([word,np.outer(word,word)[onp.triu_indices(self.N,1)]])#self.idx[0],self.idx[1]]])
    
    def calc_logp_unnormed(self,factors):
        return -self.words@factors
    
    def create_words(self):
        words = np.array(onp.fliplr(list(it.product([0,1],repeat=self.N))))
        self.words = np.array(onp.hstack([words,onp.stack([onp.outer(word,word)[onp.triu_indices(self.N,1)] for word in words])]))
    
    def calc_marginals_ex(self,words,ps):
        return self.words.T@ps


class Indep(Model):
    def __init__(self,N):
        marg_1 = lambda i,x:x[i]
        marg_1s = [jit(partial(marg_1,i)) for i in range(N)]
        super().__init__(N,funcs=marg_1s)
        self.calc_e = jit(self.calc_e)

    def calc_e(self,factors,word):
        # return np.sum(factors[:self.N]*word[:self.N]) + np.sum(factors[self.N:]*np.outer(word,word)[self.idx[0],self.idx[1]])
        return factors@word
    
    def calc_logp_unnormed(self,factors):
        return -self.words@factors
    
    def create_words(self):
        self.words = np.array(onp.fliplr(list(it.product([0,1],repeat=self.N))))

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
        # return np.sum(factors[:self.N]*word[:self.N]) + np.sum(factors[self.N:]*np.outer(word,word)[self.idx[0],self.idx[1]])
        return factors[:self.k_sync_factors_start_idx]@np.concatenate([word,np.outer(word,word)[onp.triu_indices(self.N,1)]])+factors[self.k_sync_factors_start_idx:][np.sum(word)]#self.idx[0],self.idx[1]]])
    
    def calc_logp_unnormed(self,factors):
        return -self.words@factors
    
    def create_words(self):
        words = np.array(onp.fliplr(list(it.product([0,1],repeat=self.N))))
        k_sync_idx = words.sum(1)
        k_sync = onp.zeros((words.shape[0],self.N+1))
        k_sync[onp.arange(words.shape[0]),k_sync_idx] = 1
        self.words = np.array(onp.hstack([words,onp.stack([onp.outer(word,word)[onp.triu_indices(self.N,1)] for word in words]),k_sync]))

    def calc_marginals_ex(self,words,ps):
        return self.words.T@ps
