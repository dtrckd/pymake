# -*- coding: utf-8 -*-

import collections
import logging
lgg = logging.getLogger('root')

import numpy as np
import scipy as sp

from scipy.special import digamma
from numpy.random import dirichlet, gamma, poisson, binomial, beta

from pymake.frontend.frontend import DataBase
from ml.model.modelbase import GibbsSampler
from .hdp import MSampler, BetaSampler

from pymake.util.math import lognormalize, categorical


# Implementation of Teh et al. (2005) Gibbs sampler for the Hierarchical Dirichlet Process: Direct assignement.

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

""" @Todo
* Control the calcul of perplexity at each iterations. Costly !

* HDP direct assignement:
Network implementation (2*N restaurant, symmetric, p(k,l|.) = p(k)p(l) etc)

"""

# Broadcast class based on numpy matrix
# Assume delta a scalar.
class Likelihood(object):

    def __init__(self, delta, data, **config):
        if type(data) is np.ndarray:
            # Error when streaming ? sppy ?
            #self.data_mat = sp.sparse.csr_matrix(data)
            self.data_mat = data
        elif data.format == 'csr':
            self.data_mat = data
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))
        self.data = DataBase.sparse2stream(self.data_mat)
        assert(len(self.data) > 1)
        assert(type(delta) in (int, float))
        self.delta = delta

        self.data_dims = self.get_data_dims()
        self.nnz = self.get_nnz()
        # Vocabulary size
        self.nfeat = self.get_nfeat()

        # Cst for CGS of DM and scala delta as prior.
        self.delta = delta if isinstance(delta, np.ndarray) else np.asarray([delta] * self.nfeat)
        self.w_delta = self.delta.sum()

        print(self.data_mat.shape, self.nfeat)
        assert(self.data_mat.shape[1] == self.nfeat)
        #self.data_mat = sppy.csarray(self.data_mat)

    def compute(self, j, i, k_ji):
        return self.loglikelihood(j, i, k_ji)

    def get_nfeat(self):
        return np.vectorize(max)(self.data).max() + 1

    def get_data_dims(self):
        data_dims = np.vectorize(len)(self.data)
        return list(data_dims)

    def get_nnz(self):
        return sum(self.data_dims)

    def words_k(self, where_k):
        for group in range(len(self.data)):
            word_gen = self.data[group][where_k[group]]
            for word in word_gen:
                yield word

    def make_word_topic_counts(self, z, K):
        word_topic_counts = np.zeros((self.nfeat, K), dtype=int)

        for k in range(K):
            where_k =  np.array([np.where(zj == k)[0] for zj in z])
            words_k_dict = collections.Counter(self.words_k(where_k))
            word_topic_counts[words_k_dict.keys(), k] = words_k_dict.values()

        self.word_topic_counts = word_topic_counts

    def loglikelihood(self, j, i, k_ji):
        w_ji = self.data[j][i]
        self.word_topic_counts[w_ji, k_ji] -= 1
        self.total_w_k[k_ji] -= 1
        log_smooth_count_ji = np.log(self.word_topic_counts[w_ji] + self.delta[w_ji])

        return log_smooth_count_ji - np.log(self.total_w_k + self.w_delta)

class ZSampler(object):
    # Alternative is to keep the two count matrix and
    # The docment-word topic array to trace get x(-ij) topic assignment:
        # C(word-topic)[w, k] === n_dotkw
        # C(document-topic[j, k] === n_jdotk
        # z DxN_k topic assignment matrix
        # --- From this reconstruct theta and phi

    def __init__(self, alpha_0, likelihood, K_init=0, data_t=None):
        self.K_init = K_init
        self.alpha_0 = alpha_0
        self.likelihood = likelihood
        self.data_dims = likelihood.data_dims
        self.J = len(self.data_dims)
        self.z = self._init_topics_assignement()
        self.doc_topic_counts = self.make_doc_topic_counts()
        if not hasattr(self, 'K'):
            # Nonparametric Case
            self.purge_empty_topics()
        self.likelihood.make_word_topic_counts(self.z, self.get_K())
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

        if data_t is not None:
            self.data_t = data_t
            self.data_t_w = DataBase.sparse2stream(data_t)
            self.z_t = self._init_topics_assignement(data_t)
            self.doc_topic_counts_t = self.make_doc_topic_counts(self.data_t)

        # if a tracking of topics indexis pursuit,
        # pay attention to the topic added among those purged...(A topic cannot be added and purged in the same sample iteration !)
        self.last_purged_topics = []

    # @data_dims: list of number of observations per document/instance
    def _init_topics_assignement(self, data=None):
        if data is None:
            data_dims = self.data_dims
        else:
            data_dims = self.data_t.sum(1)
        alpha_0 = self.alpha_0

        # Poisson way
        #z = np.array( [poisson(alpha_0, size=dim) for dim in data_dims] )

        # Random way
        K = self.K_init
        z = np.array( [np.random.randint(0, K, (dim)) for dim in data_dims] )

        # LDA way
        # improve local optima ?
        #todo ?

        return z

    def sample(self):
        # Add pnew container
        self._update_log_alpha_beta()
        self.doc_topic_counts =  np.column_stack((self.doc_topic_counts, np.zeros(self.J, dtype=int)))
        self.likelihood.word_topic_counts =  np.column_stack((self.likelihood.word_topic_counts, np.zeros(self.likelihood.nfeat, dtype=int)))
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

        lgg.info( 'Sample z...')
        lgg.debug( '#Doc \t nnz \t  #topic')
        doc_order = np.random.permutation(self.J)
        for doc_iter, j in enumerate(doc_order):
            nnz =  self.data_dims[j]
            lgg.debug( '%d \t %d \t %d' % ( doc_iter , nnz, self.doc_topic_counts.shape[1]-1 ))
            nnz_order = np.random.permutation(nnz)
            for i in nnz_order:
                params = self.prob_zji(j, i, self._K + 1)
                sample_topic = categorical(params)
                self.z[j][i] = sample_topic

                # Regularize matrices for new topic sampled
                if sample_topic == self.doc_topic_counts.shape[1] - 1:
                    self._K += 1
                    #print 'Simplex probabilities: %s' % (params)
                    col_doc = np.zeros((self.J, 1), dtype=int)
                    col_word = np.zeros((self.likelihood.nfeat, 1), dtype=int)
                    self.doc_topic_counts = np.hstack((self.doc_topic_counts, col_doc))
                    self.likelihood.word_topic_counts = np.hstack((self.likelihood.word_topic_counts, col_word))
                    self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

                # Update count matrixes
                self.doc_topic_counts[j, sample_topic] += 1
                self.likelihood.word_topic_counts[self.likelihood.data[j][i], sample_topic] += 1
                self.likelihood.total_w_k[sample_topic] += 1

        # Remove pnew container
        self.doc_topic_counts = self.doc_topic_counts[:, :-1]
        self.likelihood.word_topic_counts = self.likelihood.word_topic_counts[:,:-1]
        self.purge_empty_topics()

        return self.z

    def sample_heldout(self):
        J = self.data_t.shape[0]
        K = self.get_K()

        # Manage Alpha prior
        if not hasattr(self, 'logalpha'):
            alpha = np.exp(self.log_alpha_beta[:-1])
        else:
            alpha = np.exp(self.logalpha)

        doc_order = np.random.permutation(J)
        for doc_iter, j in enumerate(doc_order):
            nnz =  self.data_t[j].sum()
            lgg.debug( '%d \t %d \t %d' % ( doc_iter , nnz, K ))
            nnz_order = np.random.permutation(nnz)
            for i in nnz_order:

                k_ji = self.z_t[j][i]
                self.doc_topic_counts_t[j, k_ji] -=1

                params = np.log(self.doc_topic_counts_t[j] + alpha) + np.log(self._phi[self.data_t_w[j][i], k_ji])
                params =  lognormalize(params[:K])

                sample_topic = categorical(params)
                self.z_t[j][i] = sample_topic

                self.doc_topic_counts_t[j, sample_topic] += 1

        return self.z_t

    def make_doc_topic_counts(self, data=None):
        if data is None:
            J = self.J
            z = self.z
        else:
            J = data.shape[0]
            z = self.z_t
        K = self.get_K()
        counts = np.zeros((J, K), dtype=int)
        for j, d in enumerate(z):
            counts[j] = np.bincount(d, minlength=K)

        return counts

    def _update_log_alpha_beta(self):
        self.log_alpha_beta = np.log(self.alpha_0) + np.log(self.betasampler.beta)

    # Remove empty topic in nonparametric case
    def purge_empty_topics(self):
        counts = self.doc_topic_counts

        dummy_topics = []
        # Find empty topics
        for k, c in enumerate(counts.T):
            if c.sum() == 0:
                dummy_topics.append(k)
        for k in sorted(dummy_topics, reverse=True):
            counts = np.delete(counts, k, axis=1)
            self._K -= 1
            if hasattr(self.likelihood, 'word_topic_counts'):
                self.likelihood.word_topic_counts = np.delete(self.likelihood.word_topic_counts, k, axis=1)
            # Regularize Z
            for d in self.z:
                d[d > k] -= 1
            # Regularize alpha_beta, minus one the pnew topic
            if hasattr(self, 'log_alpha_beta') and k < len(self.log_alpha_beta)-1:
                self.log_alpha_beta = np.delete(self.log_alpha_beta, k)
                self.betasampler.beta = np.delete(self.betasampler.beta, k)

        self.last_purged_topics = dummy_topics
        if len(dummy_topics) > 0:
            lgg.info( 'zsampler: %d topics purged' % (len(dummy_topics)))
        self.doc_topic_counts =  counts

    def add_beta_sampler(self, betasampler):
        self.betasampler = betasampler
        self._update_log_alpha_beta()

    def get_K(self):
        if not hasattr(self, '_K'):
            self._K =  np.max(np.vectorize(np.max)(self.z)) + 1
        return self._K

    # Compute probabilityy to sample z_ij = k for each [K].
    # K is would be fix or +1 for nonparametric case
    def prob_zji(self, j, i, K):
        k_ji = self.z[j][i]
        self.doc_topic_counts[j, k_ji] -=1

        # Manage Alpha prior
        if not hasattr(self, 'logalpha'):
            log_alpha_beta = self.log_alpha_beta
            new_k = K - len(log_alpha_beta)
            if new_k > 0:
                log_alpha_beta = np.hstack((log_alpha_beta, np.ones((new_k,))*log_alpha_beta[-1]))
            alpha = np.exp(log_alpha_beta)
        else:
            alpha = np.exp(self.logalpha)

        params = np.log(self.doc_topic_counts[j] + alpha) + self.likelihood.compute(j, i, k_ji)
        return lognormalize(params[:K])

    def get_log_alpha_beta(self, k):
        old_max = self.log_alpha_beta.shape[0]

        if k > (old_max - 1):
            return self.log_alpha_beta[old_max - 1]
        else:
            return self.log_alpha_beta[k]

    def clean(self):
        self.K = self.doc_topic_counts.shape[1]

    def predictive_topics(self):
        # check if perplecxy is equal if removing dummy empty topics...
        if not hasattr(self, 'logalpha'):
            alpha = np.exp(self.log_alpha_beta[:-1])
        else:
            alpha = np.exp(self.logalpha)
        delta = self.likelihood.delta
        K = len(alpha)
        J = self.data_t.shape[0]

        # Recontruct Documents-Topic matrix
        _theta = self.doc_topic_counts_t + np.tile(alpha, (J, 1))
        _theta = (_theta.T / _theta.sum(axis=1)).T
        return _theta

    def estimate_latent_variables(self):
        # check if perplecxy is equal if removing dummy empty topics...
        if not hasattr(self, 'logalpha'):
            log_alpha_beta = self.log_alpha_beta
            new_k = self.get_K()+1 - len(log_alpha_beta)
            if new_k > 0:
                gmma = log_alpha_beta[-1]
                log_alpha_beta = np.hstack((log_alpha_beta, np.ones((new_k,))*gmma))
            # Remove empty possibly new topic
            alpha = np.exp(log_alpha_beta[:-1])
        else:
            alpha = np.exp(self.logalpha)
        delta = self.likelihood.delta
        K = len(alpha)

        # Recontruct Documents-Topic matrix
        _theta = self.doc_topic_counts + np.tile(alpha, (self.J, 1))
        self._theta = (_theta.T / _theta.sum(axis=1)).T

        # Recontruct Words-Topic matrix
        _phi = self.likelihood.word_topic_counts.T + np.tile(delta, (K, 1))
        self._phi = (_phi.T / _phi.sum(axis=1))

        return self._theta, self._phi

    # Mean can be arithmetic or geometric
    def perplexity(self, data=None, mean='arithmetic'):
        phi = self._phi
        if data is None:
            data = self.likelihood.data_mat
            nnz = self.likelihood.nnz
            theta = self._theta
        else:
            data = self.data_t
            nnz = self.data_t.sum()
            theta = self.predictive_topics()

        ### based on aritmetic mean

        ### Loop Approach
        #entropy = 0.0
        #_indices = lambda x: x.nonzero()[1]
        #for j in range(self.J):
        #    data_j = [ (i, data[j, i]) for i in _indices(data[j]) ]
        #    entropy += np.sum( cnt_wi * np.log(theta[j] * phi[w_ji]).sum() for w_ji, cnt_wi in data_j )

        ### Vectorized approach
        # < 2s for kos and nips k=50, quite fast
        entropy = data.multiply( np.log( theta.dot(phi.T) )).sum()
        #entropy = (data.hadamard( sppy.csarray(np.log( theta.dot(phi.T) )) )).sum()

        perplexity = np.exp(-entropy / nnz)
        return perplexity

class ZSamplerParametric(ZSampler):
    # Parametric Version of HDP sampler. Number of topics fixed.

    def __init__(self, alpha_0, likelihood, K, alpha='asymmetric', data_t=None):
        self.K = self.K_init = self._K =  K
        if 'alpha' is 'symmetric':
            alpha = np.ones(K)*1/K
        elif 'alpha' == 'asymmetric':
            alpha = np.asarray([1.0 / (i + np.sqrt(K)) for i in range(K)])
            alpha /= alpha.sum()
        else:
            alpha = np.ones(K)*alpha_0
        self.logalpha = np.log(alpha)
        super(ZSamplerParametric, self).__init__(alpha_0, likelihood, self.K, data_t=data_t)

    def sample(self):
        print( 'Sample z...')
        lgg.debug( '#Doc \t #nnz\t #Topic')
        doc_order = np.random.permutation(self.J)
        for doc_iter, j in enumerate(doc_order):
            nnz =  self.data_dims[j]
            lgg.debug( '%d \t %d \t %d' % ( doc_iter , nnz, self.K ))
            nnz_order = np.random.permutation(nnz)
            for i in nnz_order:
                params = self.prob_zji(j, i, self.K)
                sample_topic = categorical(params)
                self.z[j][i] = sample_topic

                self.doc_topic_counts[j, sample_topic] += 1
                self.likelihood.word_topic_counts[self.likelihood.data[j][i], sample_topic] += 1
                self.likelihood.total_w_k[sample_topic] += 1

        return self.z

    def get_K(self):
        return self.K

    def get_log_alpha_beta(self, k):
        return self.logalpha[k]

    def clean(self):
        pass

class NP_CGS(object):

    # Joint Sampler of topic Assignement, table configuration, and beta proportion.
    # ref to direct assignement Sampling in HDP (Teh 2006)
    def __init__(self, zsampler, msampler, betasampler, hyper='auto'):
        zsampler.add_beta_sampler(betasampler)

        self.zsampler = zsampler
        self.msampler = msampler
        self.betasampler = betasampler

        msampler.sample()
        betasampler.sample()

        if hyper.startswith( 'auto' ):
            self.hyper = hyper
            self.a_alpha = 10
            self.b_alpha = 0.2
            self.a_gmma = 10
            self.b_gmma = 0.2
            self.optimize_hyper_hdp()
        elif hyper.startswith( 'fix' ):
            self.hyper = hyper
        else:
            raise NotImplementedError('Hyperparameters optimization ?')

    def optimize_hyper_hdp(self):
        # Optimize \alpha_0
        m_dot = self.msampler.m_dotk.sum()
        alpha_0 = self.zsampler.alpha_0
        n_jdot = np.array(self.zsampler.data_dims)
        #norm = np.linalg.norm(n_jdot/alpha_0)
        #u_j = binomial(1, n_jdot/(norm* alpha_0))
        u_j = binomial(1, n_jdot/(n_jdot + alpha_0))
        v_j = beta(alpha_0 + 1, n_jdot)
        new_alpha0 = gamma(self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum()), size=5).mean()
        self.zsampler.alpha_0 = new_alpha0

        # Optimize \gamma
        K = self.zsampler._K
        gmma = self.betasampler.gmma
        #norm = np.linalg.norm(m_dot/gmma)
        #u = binomial(1, m_dot / (norm*gmma))
        u = binomial(1, m_dot / (m_dot + gmma))
        v = beta(gmma + 1, m_dot)
        new_gmma = gamma(self.a_gmma + K -1 + u, 1/(self.b_gmma - np.log(v)), size=5).mean()
        self.betasampler.gmma = new_gmma

        print('alpha a, b: %s, %s ' % (self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum())))
        print( 'hyper sample: alpha_0: %s gamma: %s' % (new_alpha0, new_gmma))
        return

    def sample(self):
            z = self.zsampler.sample()
            m = self.msampler.sample()
            beta = self.betasampler.sample()

            if self.hyper.startswith('auto'):
                self.optimize_hyper_hdp()

            if hasattr(self.zsampler, 'data_t') and self.zsampler.data_t is not None:
                self.zsampler.sample_heldout()

            return z, m, beta

class CGS(object):

    def __init__(self, zsampler):
        self.zsampler = zsampler

    def sample(self):

        z = self.zsampler.sample()

        if hasattr(self.zsampler, 'data_t') and self.zsampler.data_t is not None:
            self.zsampler.sample_heldout()

        return z

class GibbsRun(GibbsSampler):
    __abstractmethods__ = 'model'
    def __init__(self, sampler,  data_t=None, **kwargs):

        self.burnin = kwargs.get('burnin',  0.05) # Ratio of iteration
        self.thinning = kwargs.get('thinning',  1)
        self.comm = dict() # Empty dict to store communities and blockmodel structure
        self.data_t = data_t
        self._csv_typo = 'it it_time entropy_train entropy_test K alpha gamma alpha_mean delta_mean alpha_var delta_var'
        self.fmt = '%d %.4f %.8f %.8f %d %.8f %.8f %.4f %.4f %.4f %.4f'
        #self.fmt = '%s %s %s %s %s %s %s %s %s %s %s'
        GibbsSampler.__init__(self, sampler, **kwargs)


    def predict(self):
        lgg.error('todo')
        pass

