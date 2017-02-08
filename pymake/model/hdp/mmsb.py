# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import logging
lgg = logging.getLogger('root')

import numpy as np
from numpy import ma
import scipy as sp

from scipy.special import digamma
from numpy.random import dirichlet, gamma, poisson, binomial, beta

from pymake.model import GibbsSampler, MSampler, BetaSampler

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree
#from util.algo import *

# Implementation Mixed Membership Sochastic Blockmodel Stochastic

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

""" @Todo
* sparse masked array ?!@$*

* Add constant to count matrix by default will win some precious second

ref:
HDP: Teh et al. (2005) Gibbs sampler for the Hierarchical Dirichlet Process: Direct assignement.
"""

# Broadcast class based on numpy matrix
# Assume delta a scalar.
class Likelihood(object):

    def __init__(self, delta, data, nodes_list=None, assortativity=False):
        """ Notes
            -----
            * Diagonal is ignored here for prediction
        """

        if nodes_list is None:
            self.nodes_list = [np.arange(data.shape[0]), np.arange(data.shape[1])]
        else:
            self.nodes_list = nodes_list
            raise ValueError('re order the networks ! to avoid using _nmap')

        if type(data) is not np.ma.masked_array:
            # Ignore Diagonal
            data = np.ma.array(data, mask=np.zeros(data.shape))
            np.fill_diagonal(data, ma.masked)

        self.data_ma = data
        self.symmetric = (data == data.T).all()
        self.data_dims = self.get_data_dims()
        self.nnz = self.get_nnz()
        # Vocabulary size
        self.nfeat = self.get_nfeat()

        # assert for coocurence matric
        #assert(self.data_mat.shape[1] == self.nfeat)
        #self.data_mat = sppy.csarray(self.data_mat)

        # Cst for CGS of DM and scala delta as prior.
        self.assortativity = assortativity
        self.delta = np.asarray(delta) if isinstance(delta, (np.ndarray, list, tuple)) else np.asarray([delta] * self.nfeat)
        self.w_delta = self.delta.sum()
        if assortativity == 1:
            raise NotImplementedError('assort 2')
        elif assortativity == 2:
            self.epsilon = 0.01
        else:
            pass

        # for loglikelihood bernoulli computation
        self.data_A = self.data_ma.copy()
        self.data_A.data[self.data_A.data == 0] = -1
        self.data_B = np.ones(self.data_ma.shape) - self.data_ma


    def compute(self, j, i, k_ji, k_ij):
        return self.loglikelihood(j, i, k_ji, k_ij)

    def get_nfeat(self):
        nfeat = self.data_ma.max() + 1
        if nfeat == 1:
            lgg.warn( 'Warning, only zeros in adjacency matrix...')
            nfeat = 2
        return nfeat

    # Contains the index of nodes with who it interact.
    # @debug no more true for bipartite networks
    def get_data_dims(self):
        #data_dims = np.vectorize(len)(self.data)
        #data_dims = [r.count() for r in self.data_ma]
        data_dims = []
        for i in range(self.data_ma.shape[0]):
            data_dims.append(self.data_ma[i,:].count() + self.data_ma[:,i].count())
        return data_dims

    def get_nnz(self):
        return len(self.data_ma.compressed())

    # Need it in the case of sampling sysmetric networks. (only case where ineed to map ?)
    # return the true node index corresponding to arbitrary index i of matrix count/data position
    # @pos: 0 indicate line picking, 1 indicate rows picking
    def _nmap(self, i, pos):
        return self.nodes_list[pos][i]

    # @debug: symmetric matrix ?
    def make_word_topic_counts(self, z, K):
        word_topic_counts = np.zeros((self.nfeat, K, K), dtype=int)

        for j, i in self.data_iter():
            z_ji = z[j,i,0]
            z_ij = z[j,i,1]
            word_topic_counts[self.data_ma[j, i], z_ji, z_ij] += 1
            if self.symmetric:
                word_topic_counts[self.data_ma[j, i], z_ij, z_ji] += 1

        self.word_topic_counts = word_topic_counts

    # Interface to properly iterate over data
    def data_iter(self, randomize=True):
        if not hasattr(self, '_order'):
            order = np.arange(len(self.data_dims)**2).reshape(self.data_ma.shape)
            masked = order[self.data_ma.mask]

            if self.symmetric:
                tril = np.tril_indices_from(self.data_ma, -1)
                tril = order[tril]
                masked =  np.append(masked, tril)

            # Remove masked value to the iteration list
            order = np.delete(order, masked)
            # Get the indexes of nodes (i,j) for each observed interactions
            order = list(zip(*np.unravel_index(order, self.data_ma.shape)))
            self._order = order
        else:
            order = self._order

        if randomize is True:
            np.random.shuffle(order)
        return order

    # @debug: symmetric matrix ?
    def loglikelihood(self, j, i, k_ji, k_ij):
        w_ji = self.data_ma[j, i]
        self.word_topic_counts[w_ji, k_ji, k_ij] -= 1
        self.total_w_k[k_ji, k_ij] -= 1
        if self.symmetric:
            self.word_topic_counts[w_ji, k_ij, k_ji] -= 1
            self.total_w_k[k_ij, k_ji] -= 1

        if self.assortativity == 2:
            if k_ji == k_ij:
                log_smooth_count_ji = np.log(self.word_topic_counts[w_ji] + self.delta[w_ji])
                ll = log_smooth_count_ji - np.log(self.total_w_k + self.w_delta)
            else:
                ll = self.epsilon

        else:
            log_smooth_count_ji = np.log(self.word_topic_counts[w_ji] + self.delta[w_ji])
            ll = log_smooth_count_ji - np.log(self.total_w_k + self.w_delta)

        return ll


class ZSampler(object):
    # Alternative is to keep the two count matrix and
    # The docment-word topic array to trace get x(-ij) topic assignment:
        # C(word-topic)[w, k] === n_dotkw
        # C(document-topic[j, k] === n_jdotk
        # z DxN_k topic assignment matrix
        # --- From this reconstruct theta and phi

    def __init__(self, alpha_0, likelihood, K_init=1, data_t=None):
        self.K_init = K_init or 1
        self.alpha_0 = alpha_0
        self.likelihood = likelihood
        self.symmetric_pt = (self.likelihood.symmetric&1) +1 # the increment for Gibbs iteration
        self._nmap = likelihood._nmap
        self.nodes_list = likelihood.nodes_list
        self.data_dims = self.likelihood.data_dims
        self.J = len(self.data_dims)
        self.z = self._init_topics_assignement()
        self.doc_topic_counts = self.make_doc_topic_counts()
        if not hasattr(self, 'K'):
            # Nonparametric Case
            self.purge_empty_topics()
        self.likelihood.make_word_topic_counts(self.z, self.get_K())
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

        # if a tracking of topics indexis pursuit,
        # pay attention to the topic added among those purged...(A topic cannot be added and purged in the same sample iteration !)
        self.last_purged_topics = []

    # @debug: symmetric matrix ?
    def _init_topics_assignement(self):
        dim = (self.J, self.J, 2)
        alpha_0 = self.alpha_0

        # Poisson way
        #z = np.array( [poisson(alpha_0, size=dim) for dim in data_dims] )

        # Random way
        K = self.K_init
        z = np.random.randint(0, K, (dim))

        if self.likelihood.symmetric:
            z[:, :, 0] = np.triu(z[:, :, 0]) + np.triu(z[:, :, 0], 1).T
            z[:, :, 1] = np.triu(z[:, :, 1]) + np.triu(z[:, :, 1], 1).T

        # LDA way
        # improve local optima ?
        #theta_j = dirichlet([1, gmma])
        #todo ?

        return z

    def sample(self):
        # Add pnew container
        self._update_log_alpha_beta()
        self.update_matrix_shape()

        lgg.info('Sample z...')
        lgg.debug('#J \t #I \t  #topic')
        doc_order = np.random.permutation(self.J)
        # @debug: symmetric matrix !
        for j, i in self.likelihood.data_iter(randomize=True):
            lgg.debug( '%d \t %d \t %d' % ( j , i, self.doc_topic_counts.shape[1]-1))
            params = self.prob_zji(j, i, self._K + 1)
            sample_topic_raveled = categorical(params)
            k_j, k_i = np.unravel_index(sample_topic_raveled, (self._K+1, self._K+1))
            k_j, k_i = k_j[0], k_i[0] # beurk :(
            self.z[j, i, 0] = k_j
            self.z[j, i, 1] = k_i

            # Regularize matrices for new topic sampled
            if k_j == self.doc_topic_counts.shape[1] - 1 or k_i == self.doc_topic_counts.shape[1] - 1:
                self._K += 1
                #print 'Simplex probabilities: %s' % (params)
                self.update_matrix_shape(new_topic=True)

            self.update_matrix_count(j,i,k_j, k_i)

        # Remove pnew container
        self.doc_topic_counts = self.doc_topic_counts[:, :-1]
        self.likelihood.word_topic_counts = self.likelihood.word_topic_counts[:, :-1, :-1]
        self.purge_empty_topics()

        return self.z

    def update_matrix_shape(self, new_topic=False):
        if new_topic is True:
            # Updata alpha
            self.log_alpha_beta = np.append(self.log_alpha_beta, self.log_alpha_beta[-1])
            self.alpha = np.append(self.alpha, np.exp(self.log_alpha_beta[-1]))

        # Update Doc-topic count
        new_inst = np.zeros((self.J, 1), dtype=int)
        self.doc_topic_counts = np.hstack((self.doc_topic_counts, new_inst))

        # Update word-topic count
        new_feat_1 = np.zeros((self.likelihood.nfeat, self._K), dtype=int)
        new_feat_2 = np.zeros((self.likelihood.nfeat, self._K+1), dtype=int)
        self.likelihood.word_topic_counts = np.dstack((self.likelihood.word_topic_counts, new_feat_1))
        self.likelihood.word_topic_counts = np.hstack((self.likelihood.word_topic_counts, new_feat_2[:, None]))
        # sum all to update to fit the shape (a bit nasty if operation (new topic) occur a lot)
        self.likelihood.total_w_k = self.likelihood.word_topic_counts.sum(0)

    def update_matrix_count(self, j, i, k_j, k_i):
        self.doc_topic_counts[j, k_j] += self.symmetric_pt
        self.doc_topic_counts[i, k_i] += self.symmetric_pt
        self.likelihood.word_topic_counts[self.likelihood.data_ma[j,i], k_j, k_i] += 1
        self.likelihood.total_w_k[k_j, k_i] += 1
        if self.likelihood.symmetric:
            self.likelihood.word_topic_counts[self.likelihood.data_ma[j,i], k_i, k_j] += 1
            self.likelihood.total_w_k[k_i, k_j] += 1

    # @debug: symmetric matrix ?
    def make_doc_topic_counts(self):
        K = self.get_K()
        counts = np.zeros((self.J, K), dtype=int)

        for j, i in self.likelihood.data_iter(randomize=False):
            k_j = self.z[j, i, 0]
            k_i = self.z[j, i, 1]
            counts[j, k_j] += self.symmetric_pt
            counts[i, k_i] += self.symmetric_pt
        return counts

    def _update_log_alpha_beta(self):
        self.log_alpha_beta = np.log(self.alpha_0) + np.log(self.betasampler.beta)
        self.alpha = np.exp(self.log_alpha_beta)

    # Remove empty topic in nonparametric case
    # @debug: symmetric matrix ?
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
                self.likelihood.word_topic_counts = np.delete(self.likelihood.word_topic_counts, k, axis=2)
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
            self._K =  np.max(self.z) + 1
        return self._K

    # Compute probabilityy to sample z_ij = k for each [K].
    # K is would be fix or +1 for nonparametric case
    def prob_zji(self, j, i, K):
        k_jji = self.z[j, i, 0]
        k_jij = self.z[j, i, 1]
        self.doc_topic_counts[j, k_jji] -= self.symmetric_pt
        self.doc_topic_counts[i, k_jij] -= self.symmetric_pt

        # Keep the outer product in memory
        p_jk = self.doc_topic_counts[j] + self.alpha
        p_ik = self.doc_topic_counts[i] + self.alpha
        outer_kk = np.outer(p_jk, p_ik)

        params = np.log(outer_kk) + self.likelihood.compute(j, i, k_jji, k_jij)
        params = params[:K, :K].ravel()
        return lognormalize(params)

    def get_log_alpha_beta(self, k):
        old_max = self.log_alpha_beta.shape[0]

        if k > (old_max - 1):
            return self.log_alpha_beta[old_max - 1]
        else:
            return self.log_alpha_beta[k]

    def clean(self):
        self.K = self.doc_topic_counts.shape[1]

    def predictive_topics(self, data):
        pass

    def estimate_latent_variables(self):
        # check if perplexity is equal if removing dummy empty topics...
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
        _phi = self.likelihood.word_topic_counts + np.tile(delta, (K, K, 1)).T
        self._phi = (_phi / np.linalg.norm(_phi, axis=0))[1]

        return self._theta, self._phi

    # Mean can be arithmetic or geometric
    def perplexity(self, data=None, mean='arithmetic'):
        phi = self._phi
        if data is None:
            data = self.likelihood.data_ma
            nnz = self.likelihood.nnz
            theta = self._theta
        else:
            nnz = data.sum()
            theta = self.predictive_topics(data)

        ### based on aritmetic mean

        ### Loop Approach
        #entropy = 0.0
        #_indices = lambda x: x.nonzero()[1]
        #for j in xrange(self.J):
        #    data_j = [ (i, data[j, i]) for i in _indices(data[j]) ]
        #    entropy += np.sum( cnt_wi * np.log(theta[j] * phi[w_ji]).sum() for w_ji, cnt_wi in data_j )

        ### Vectorized approach
        # < 2s for kos and nips k=50, quite fast
        p_ji =  theta.dot(phi).dot(theta.T)
        # p_ji if a links, (1 - p_ji) if not a links
        p_ji = self.likelihood.data_A * p_ji + self.likelihood.data_B

        # Commented because count matrices are kept symmetric. So normalized over all to keep coherence.
        #if self.likelihood.symmetric:
        #    entropy =  np.log(p_ji[self.likelihood.triu]).sum()
        #else:
        #    entropy =  np.log(p_ji).sum()
        entropy =  np.log(p_ji).sum()

        #perplexity = np.exp(-entropy / nnz)
        entropy = - entropy / nnz
        return entropy

class ZSamplerParametric(ZSampler):
    # Parametric Version of HDP sampler. Number of topics fixed.

    def __init__(self, alpha_0, likelihood, K, alpha='asymmetric', data_t=None):
        self.K = self.K_init = self._K =  K
        if 'alpha' in ('symmetric', 'fix'):
            alpha = np.ones(K)*1/K
        elif 'alpha' in ('asymmetric', 'auto'):
            alpha = np.asarray([1.0 / (i + np.sqrt(K)) for i in xrange(K)])
            alpha /= alpha.sum()
        else:
            alpha = np.ones(K)*alpha_0
        self.logalpha = np.log(alpha)
        self.alpha = alpha
        super(ZSamplerParametric, self).__init__(alpha_0, likelihood, self.K, data_t=data_t)

    def sample(self):
        lgg.info('Sample z...')
        lgg.debug('#J \t #I \t #topic')
        doc_order = np.random.permutation(self.J)
        # @debug: symmetric matrix !
        for j, i in self.likelihood.data_iter(randomize=True):
            lgg.debug( '%d \t %d \t %d' % (j , i, self.doc_topic_counts.shape[1]-1))
            params = self.prob_zji(j, i, self.K)
            sample_topic_raveled = categorical(params)
            k_j, k_i = np.unravel_index(sample_topic_raveled, (self._K, self._K))
            k_j, k_i = k_j[0], k_i[0] # beurk :(
            self.z[j, i, 0] = k_j
            self.z[j, i, 1] = k_i
            nodes_classes_ass = [(j, k_j), (i, k_i)]

            self.update_matrix_count(j, i, k_j, k_i)
        return self.z

    def get_K(self):
        return self.K

    def get_log_alpha_beta(self, k):
        return self.logalpha[k]

    def clean(self):
        pass


class NP_CGS(object):

    # Joint Sampler of topic Assignement, table configuration, and beta proportion.
    # ref to direct assignement Sampling in HDP (Teh 2006)
    def __init__(self, zsampler, msampler, betasampler, hyper='auto', hyper_prior=None):
        zsampler.add_beta_sampler(betasampler)

        self.zsampler = zsampler
        self.msampler = msampler
        self.betasampler = betasampler

        msampler.sample()
        betasampler.sample()

        if hyper.startswith( 'auto' ):
            self.hyper = hyper
            if hyper_prior is None:
                self.a_alpha = 1
                self.b_alpha = 1
                self.a_gmma = 1
                self.b_gmma = 1
            else:
                self.a_alpha = hyper_prior[0]
                self.b_alpha = hyper_prior[1]
                self.a_gmma = hyper_prior[2]
                self.b_gmma = hyper_prior[3]

            self.optimize_hyper_hdp()
        elif hyper.startswith( 'fix' ):
            self.hyper = hyper
        else:
            raise NotImplementedError('Hyperparameters optimization ?')

    def optimize_hyper_hdp(self):
        # Optimize \alpha_0
        m_dot = self.msampler.m_dotk.sum()
        alpha_0 = self.zsampler.alpha_0
        n_jdot = np.array(self.zsampler.data_dims) # @debug add row count + line count for masked !
        #p = np.power(n_jdot / alpha_0, np.arange(n_jdot.shape[0]))
        #norm = np.linalg.norm(p)
        #u_j = binomial(1, p/norm)
        u_j = binomial(1, alpha_0/(n_jdot + alpha_0))
        #u_j = binomial(1, n_jdot/(n_jdot + alpha_0))
        v_j = beta(alpha_0 + 1, n_jdot)
        new_alpha0 = gamma(self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum()), size=3).mean()
        self.zsampler.alpha_0 = new_alpha0

        # Optimize \gamma
        K = self.zsampler._K
        #m_dot = self.msampler.m_dotk
        gmma = self.betasampler.gmma
        #p = np.power(m_dot / gmma, np.arange(m_dot.shape[0]))
        #norm = np.linalg.norm(p)
        #u = binomial(1, p/norm)
        u = binomial(1, gmma / (m_dot + gmma))
        #u = binomial(1, m_dot / (m_dot + gmma))
        v = beta(gmma + 1, m_dot)
        new_gmma = gamma(self.a_gmma + K -1 + u, 1/(self.b_gmma - np.log(v)), size=3).mean()
        self.betasampler.gmma = new_gmma

        #print 'm_dot %d, alpha a, b: %s, %s ' % (m_dot, self.a_alpha + m_dot - u_j.sum(), 1/( self.b_alpha - np.log(v_j).sum()))
        #print 'gamma a, b: %s, %s ' % (self.a_gmma + K -1 + u, 1/(self.b_gmma - np.log(v)))
        lgg.info('hyper sample: alpha_0: %s gamma: %s' % (new_alpha0, new_gmma))
        return

    def sample(self):
            z = self.zsampler.sample()
            m = self.msampler.sample()
            beta = self.betasampler.sample()

            if self.hyper.startswith('auto'):
                self.optimize_hyper_hdp()

            return z, m, beta

class CGS(object):

    def __init__(self, zsampler):
        self.zsampler = zsampler

    def sample(self):
        return self.zsampler.sample()

class GibbsRun(GibbsSampler):

    def __init__(self, sampler,  data_t=None, **kwargs):
        self.burnin = kwargs.get('burnin',  0.05) # Ratio of iteration
        self.thinning = kwargs.get('thinning',  1)
        self.comm = dict() # Empty dict to store communities and blockmodel structure
        self.data_t = data_t
        self.csv_typo = '# it it_time entropy_train entropy_test K alpha gamma alpha_mean delta_mean alpha_var delta_var'
        self.fmt = '%d %.4f %.8f %.8f %d %.8f %.8f %.4f %.4f %.4f %.4f'
        #self.fmt = '%s %s %s %s %s %s %s %s %s %s %s'
        GibbsSampler.__init__(self, sampler, **kwargs)

        self.mask = self.s.zsampler.likelihood.data_ma.mask

    def limit_k(self, N, directed=True):
        alpha, gmma, delta = self.get_hyper()
        N = int(N)
        if directed is True:
            m = alpha * N * (digamma(N+alpha) - digamma(alpha))
        else:
            m = alpha * N * (digamma(2*N+alpha) - digamma(alpha))

        # Number of class in the CRF
        K = int(gmma * (digamma(m+gmma) - digamma(gmma)))
        return K

    def generate(self, N, K=None, hyper=None, mode='predictive', directed=True):
        self.update_hyper(hyper)
        alpha, gmma, delta = self.get_hyper()
        N = int(N)
        if mode == 'evidence' :
            if type(self.s) is NP_CGS:
                # @todo: compute the variance for random simulation
                # Number of table in the CRF
                if directed is True:
                    m = alpha * N * (digamma(N+alpha) - digamma(alpha))
                else:
                    m = alpha * N * (digamma(2*N+alpha) - digamma(alpha))

                # Number of class in the CRF
                K = int(gmma * (digamma(m+gmma) - digamma(gmma)))
                alpha = gem(gmma, K)
                i = 0
                while i<3:
                    try:
                        dirichlet(alpha, size=N)
                        i=0
                        break
                    except ZeroDivisionError:
                        # Sometimes umprobable values !
                        alpha = gem(gmma, K)
                        i += 1
            else:
                K = int(K)
                alpha = np.ones(K) * alpha
                ##alpha = np.asarray([1.0 / (i + np.sqrt(K)) for i in xrange(K)])
                #alpha /= alpha.sum()
            #delta = self.s.zsampler.likelihood.delta

            # Generate Theta
            if i > 0:
                params, order = zip(*np.sorted(zip(alpha, range(len(alpha)), reverse=True)))
                _K = int(1/3. * len(alpha))
                alpha[order[:_K]] = 1
                alpha[order[_K:]] = 0
                theta = multinomial(1, alpha, size=N)
            else:
                theta = dirichlet(alpha, size=N)
            # Generate Phi
            phi = beta(delta[0], delta[1], size=(K,K))
            if directed is True:
                phi = np.triu(phi) + np.triu(phi, 1).T

        elif mode == 'predictive':
            theta, phi = self.reduce_latent()
            K = theta.shape[1]

        Y = np.empty((N,N))
        #pij[pij >= 0.5 ] = 1
        #pij[pij < 0.5 ] = 0
        #Y = pij
        pij = self.likelihood(theta, phi)
        pij = np.clip(pij, 0, 1)
        Y = sp.stats.bernoulli.rvs(pij)

        #for j in xrange(N):
        #    print 'j %d' % j
        #    for i in xrange(N):
        #        zj = categorical(theta[j])
        #        zi = categorical(theta[i])
        #        Y[j, i] = sp.stats.bernoulli.rvs(B[zj, zi])
        self.theta = theta
        self.phi = phi
        self.K = K
        return Y, theta, phi

    def mask_probas(self, data):
        mask = self.get_mask()
        y_test = data[mask]
        theta, phi = self.reduce_latent()
        p_ji = theta.dot(phi).dot(theta.T)
        probas = p_ji[mask]
        return y_test, probas

    def get_mask(self):
        return self.mask

    def getK(self):
        return self.s.zsampler._K


    def get_clusters(self, K=None, skip=0):
        """ Return a vector of clusters membership of nodes.

        Parameters
        ----------
        K : int
          Number of clusters. if None, used the number learn at inference.
        below: int
          skip the x firt bigger class.
        """
        theta, phi = self.get_params()
        clusters = np.argmax(theta.dot(phi), axis=1)
        if not K: return clusters
        hist = np.bincount(clusters)
        sorted_clusters = sorted(zip(hist, range(len(hist))), reverse=True)[skip:K+skip]
        _, strong_c = zip(*sorted_clusters)
        mask = np.ones(theta.shape)
        mask[:, strong_c] = 0
        theta_ma = ma.array(theta, mask=mask)
        clusters = np.argmax(theta_ma, axis=1)
        return clusters

    #add sim optionsin clusters
    def get_communities(self, K=None, skip=0):
        """ Return a vector of clusters membership of nodes.

        Parameters
        ----------
        K : int
          Number of clusters. if None, used the number learn at inference.
        """
        theta, phi = self.get_params()
        K = K or theta.shape[1]

        # Kmeans on Similarity
        _phi = phi.copy()
        _phi[phi < phi.mean()] = 0
        clusters = kmeans(theta.dot(phi), K=K)
        #s = sorted(zip(phi.ravel(), np.arange(phi.size)), reverse=True)[:K]
        #strong_c  = np.unravel_index(zip(*s)[1])
        #sim = theta.dot(phi).dot(theta.T)

        # Strongest communities
        #hist = phi.diagonal()
        #sorted_clusters = sorted(zip(hist, range(len(hist))), reverse=True)[skip:K+skip]
        #_, strong_c = zip(*sorted_clusters)
        #mask = np.ones(theta.shape)
        #mask[:, strong_c] = 0
        #theta_ma = ma.array(theta, mask=mask)
        #clusters = np.argmax(theta_ma, axis=1)

        return clusters

    #@wrapper !
    ### Degree by class (maxassignment)
    def communities_analysis(self, data, clustering='modularity'):
        symmetric = (data == data.T).all()

        if hasattr(self, clustering):
            f = getattr(self, clustering)
        else:
            raise NotImplementedError('Clustering algorithm unknow')

        if not hasattr(self, 'comm'):
            self.comm = dict()

        self.comm.update( f(data=data, symmetric=symmetric) )
        return self.comm

    def max_assignement(self, **kwargs):
        data = kwargs['data']
        symmetric = kwargs['symmetric']

        clusters = self.get_clusters()
        block_hist = np.bincount(clusters)

        local_degree = {}
        for n, c in enumerate(clusters):
            comm = str(c)
            local = local_degree.get(comm, [])
            degree_n = data[n,:].sum()
            #degree_n = data[n,:][clusters == c].sum()
            if not symmetric:
                degree_n += data[:, n].sum()
                #degree_n += data[:, n][clusters == c].sum()
            local.append(degree_n)
            local_degree[comm] = local

        return {'local_degree':local_degree,
                'clusters':clusters,
                'block_hist': block_hist,
                'size': len(block_hist)}

    ### Degree by class (sitll max assigment !)
    def modularity(self, **kwargs):
        data = kwargs['data']
        symmetric = kwargs['symmetric']

        clusters = self.get_clusters()
        block_hist = np.bincount(clusters)

        local_degree = {}
        if symmetric:
            k_perm = np.unique( list(map(list, map(set, itertools.product(np.unique(clusters) , repeat=2)))))
        else:
            k_perm = itertools.product(np.unique(clusters) , repeat=2)

        for c in k_perm:
            if type(c) in (np.float64, np.int64):
                # one clusters (as it appears for real with max assignment
                l = k = c
            elif  len(c) == 2:
                # Stochastic Equivalence (extra class bind
                k, l = c
            else:
                # Comunnities (intra class bind)
                k = l = c.pop()
            comm = '-'.join([str(k), str(l)])
            local = local_degree.get(comm, [])

            C = np.tile(clusters, (data.shape[0],1))
            y_c = data * ((C==k) & (C.T==l))
            if y_c.size > 0:
                local_degree[comm] = list(adj_to_degree(y_c).values())

            # Summing False !
            #for n in np.arange(data.shape[0]))[clusters == k]:
            #    degree_n = data[n,:][(clusters == k) == (clusters == l)].sum()
            #    if not symmetric:
            #        degree_n = data[n,:][(clusters == k) == (clusters == l)].sum()
            #    local.append(degree_n)
            #local_degree[comm] = local

        return {'local_degree':local_degree,
                'clusters':clusters,
                'block_hist': block_hist,
                'size': len(block_hist)}

    def blockmodel_ties(self, data, remove_empty=True):
        """ return ties based on a weighted average
            of the local degree ditribution """

        if not hasattr(self, 'comm'):
            self.communities_analysis(data)
        comm = self.comm

        m = [ wmean(a[0], a[1], mean='arithmetic')  for d in comm['local_degree'].values() for a in [degree_hist(d)] if len(a[0])>0 ]
        # Mean
        #m = map(np.mean, comm['local_degree'].values())
        # Variance Unused How to represant that !?
        #v = np.array(map(np.std, comm['local_degree'].values()))

        # factorize by using clusters hist instead
        hist, label = sorted_perm(m, comm['local_degree'].keys(), reverse=True)

        if remove_empty is True:
            null_classes = (hist == 0).sum()
            if null_classes > 0:
                hist = hist[:-null_classes]; label = label[:-null_classes]

        bm = zip(label, hist)
        self.comm['block_ties'] = bm
        return bm


