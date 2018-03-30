# -*- coding: utf-8 -*-

from time import time
import itertools
import logging
lgg = logging.getLogger('root')

import numpy as np
import scipy as sp
from numpy import ma
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
sp_dot = csr_matrix.dot

from .ibp import IBP
from ml.model.modelbase import GibbsSampler
from pymake.util.algo import *

# We will be taking log(0) = -Inf, so turn off this warning
#np.seterr(divide='ignore')

"""
Implements MCMC inference for the Infinite Latent Feature Relationnal Model [1].
This code was modified from the code originally written by Zhai Ke (kzhai@umd.edu).

[1] Kurt Miller, Michael I Jordan, and Thomas L Griffiths. Nonparametric latent feature models for link prediction. In Advances in neural information processing systems 2009.
"""

### @TODO
#   * 2 parameter ibp
#   * poisson-gamma ibp for real valued relation
#   * Oprimization:
#   *   - updating only the subset of relation affected by feature modified (active in both sides).
#   * Structure:
#       * Sample methods, divide and conquer
#

W_diag = -2

class IBPGibbsSampling(IBP, GibbsSampler):
    __abstractmethods__ = 'model'
    def __init__(self, expe, frontend,
                 assortativity=False,
                 alpha_hyper_parameter=None,
                 sigma_w_hyper_parameter=None,
                 metropolis_hastings_k_new=True,):
        self._sigma_w_hyper_parameter = sigma_w_hyper_parameter
        self.bilinear_matrix = None
        self.log_likelihood = None
        self.assortativity = assortativity
        self._overflow = 1.0
        self.ratio_MH_F = 0.0
        self.ratio_MH_W = 0.0
        self.snapshot_freq = 20

        self.burnin = expe.get('burnin',  5) # (inverse burnin, last sample to keep
        self.thinning = expe.get('thinning',  1)
        self._csv_typo = '_iteration time_it _entropy _entropy_t _K _alpha _sigma_w Z_sum ratio_MH_F ratio_MH_W'
        #self._fmt = '%d %.4f %.8f %.8f %d %.8f %.8f %d %.4f %.4f'
        IBP.__init__(self, alpha_hyper_parameter, metropolis_hastings_k_new)
        GibbsSampler.__init__(self, expe, frontend)
    """
    @param data: a NxD np data matrix
    @param alpha: IBP hyper parameter
    @param sigma_w: standard derivation of the feature
    @param initialize_Z: seeded Z matrix """
    def _initialize(self, frontend, alpha=1.0, sigma_w=1,
                    initial_Z=None, initial_W=None, KK=None):

        self._mean_w = 0
        assert(type(sigma_w) is float)
        self._sigma_w = sigma_w
        self._sigb = 1 # Carreful make overflow in exp of sigmoid !

        if frontend is None:
            return

        self.mask = frontend.data_ma.mask

        self.symmetric = frontend.is_symmetric()
        self.nnz = len(frontend.data_ma.compressed())
        super(IBPGibbsSampling, self)._initialize(frontend, alpha, initial_Z, KK=KK)

        self._W_prior = np.zeros((1, self._K))
        if initial_W != None:
            self._W = initial_W
        else:
            if self.assortativity == 1:
                # Identity
                self._W  = (np.ones((self._K, self._K))*W_diag) * (np.ones((self._K)) + np.eye(self._K)*-2)
            elif self.assortativity == 2:
                # Bivariate Gaussian
                v = 10
                x, y = np.mgrid[-v:v:self._K*1j, -v:v:self._K*1j]
                xy = np.column_stack([x.flat, y.flat])
                mu = np.array([0, 0])
                sigma = np.array([1, 1])
                covariance = np.array([[v*100,0],[0,v/10]])
                theta = np.pi / 4
                rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                covariance = rot.dot(covariance).dot(rot.T)
                z = sp.stats.multivariate_normal.pdf(xy, mean=mu, cov=covariance)
                z = 400 * z.reshape(x.shape)

                self.z_mean = z - np.ones(z.shape)*1
                self._W = np.random.normal(self.z_mean, self._sigma_w, (self._K, self._K))
            else:
                self._W = np.random.normal(self._mean_w, self._sigma_w, (self._K, self._K))

            if self.symmetric:
                self._W = np.tril(self._W) + np.tril(self._W, -1).T
                np.fill_diagonal(self._W, 1)

        #self._Z = csr_matrix(self._Z)
        #self._Z = lil_matrix(self._Z)

        assert(self._W.shape == (self._K, self._K))

    def fit(self, *args, **kwargs):
        # Two things to merge unify with GibbsSample !!!
        #   * rename parameters _Z and _W ot _theta and _Phi
        #   * appand all the samplee in self.s to factorize the method self.sample

        self._init()

        lgg.info( '__init__  Entropy: %f' % (-self.log_likelihood_Y() / self.nnz))
        for _it in range(self.iterations):
            self._iteration = _it

            begin_it = time()
            self.sample()

            if _it >= self.iterations - self.burnin:
                if _it % self.thinning == 0:
                    self.samples.append([self._Z, self._W])

            self.compute_measures(begin_it)
            lgg.info("iteration: %i,  Entropy : %f \t\t K=%i,  Entropy Z: %f, alpha: %f sigma_w: %f Z.sum(): %i" % (_it, self._entropy, self._K, self._entropy_Z, self._alpha, self._sigma_w, self.Z_sum))
            if self._write:
                self.write_current_state(self)
                if _it > 0 and _it % self.snapshot_freq == 0:
                    self.save(silent=True)

        ### Clean Things
        if not self.samples:
            self.samples.append([self._Z, self._W])
        self._reduce_latent()
        self.samples = None # free space

        Yd = self._Y.data
        Yd[Yd <= 0 ] = 0
        Yd[Yd > 0 ] = 1

        return

    def compute_measures(self, begin_it=0):

        ### Output / Measures
        self._entropy = self.log_likelihood
        self._entropy_t = None
        self._entropy_Z = np.nan # self.log_likelihood_Z()
        self.Z_sum = (self._Z == 1).sum()

        self.time_it = time() - begin_it

    def sample(self):

        # Can't get why I need this !
        self.log_likelihood_Y()
        # Sample every object
        order = np.random.permutation(self._N)
        for (object_counter, object_index) in enumerate(order):
            #sys.stdout.write('Z')
            singleton_features = self.sample_Zn(object_index)

            if self._metropolis_hastings_k_new:
                if self.metropolis_hastings_K_new(object_index, singleton_features):
                    #sys.stdout.write('Z+')
                    self.ratio_MH_F += 1
            #sys.stdout.flush()

        self.ratio_MH_F /= len(order)

        # Regularize matrices
        self.regularize_matrices()

        if self.assortativity == 1:
            self._W  = (np.ones((self._K, self._K))*W_diag) * (np.ones((self._K)) + np.eye(self._K)*-2)
            #self._W  = np.eye(self._K)
        elif self.assortativity == 2:
            self.sample_W()
        else:
            self.sample_W()

        if self._alpha_hyper_parameter:
            self._alpha = self.sample_alpha()

        if self._sigma_w_hyper_parameter != None:
            self._sigma_w = self.sample_sigma_w(self._sigma_w_hyper_parameter)

    def sample_Zn(self, object_index):
        ''' :param: object_index: an int data type, indicates the object index (row index) of Z we want to sample '''

        assert(type(object_index) == int or type(object_index) == np.int32 or type(object_index) == np.int64)

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(np.float)

        #m = np.array(m).reshape(-1)
        #new_m = np.array(new_m).reshape(-1)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z0 = np.log(1.0 - new_m / self._N)
        log_prob_z1 = np.log(new_m / self._N)
        log_prob_z = {0: log_prob_z0, 1: log_prob_z1}

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]

        order = np.random.permutation(self._K)
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                old_znk = self._Z[object_index, feature_index]
                new_znk = 1 if old_znk == 0 else 0

                # compute the log likelihood when Znk = old
                #log_old = self.log_likelihood_Y(object_index=(object_index, feature_index))
                log_old = self.log_likelihood
                bilinear = self.bilinear_matrix.copy()
                #log_old_1 = self.log_likelihood_Y(object_index=(object_index, feature_index))
                #if (not (bilinear == self.bilinear_matrix).all()):
                #    print feature_counter, object_index, feature_index
                #    print np.where(self.bilinear_matrix != bilinear)
                self._overflow = - log_old -1
                prob_old = log_old + log_prob_z[old_znk][feature_index]

                # compute the log likelihood when Znk = new
                self._Z[object_index, feature_index] = new_znk
                log_new = self.log_likelihood_Y(object_index=(object_index, feature_index))
                if log_new > -self._overflow:
                    self._overflow = - log_new -1
                prob_new = log_new + log_prob_z[new_znk][feature_index]

                prob_new = np.exp(prob_new + self._overflow)
                prob_old = np.exp(prob_old + self._overflow)
                Znk_is_new = prob_new / (prob_new + prob_old)
                if Znk_is_new > 0 and np.random.random() <= Znk_is_new:
                    # gibbs_accept ++
                    pass
                else:
                    self._Z[object_index, feature_index] = old_znk
                    self.bilinear_matrix = bilinear
                    self.log_likelihood = log_old

        return singleton_features

    """
    sample K_new using metropolis hastings algorithm """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index]

        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = sp.stats.poisson.rvs(self._alpha / self._N)
        K_new = self._K + K_temp - len(singleton_features)

        if K_new <= 0 or K_temp <= 0 and len(singleton_features) <= 0:
            return False

        # generate new weight from a normal distribution with mean 0 and variance sigma_w, a K_new-by-K matrix
        non_singleton_features = [k for k in range(self._K) if k not in singleton_features]
        W_temp_v = np.random.normal(self._mean_w, self._sigma_w, (K_temp, self._K - len(singleton_features)))
        W_temp_h = np.random.normal(self._mean_w, self._sigma_w, (K_new, K_temp))
        W_new = np.delete(self._W, singleton_features,0)
        W_new = np.vstack((np.delete(W_new, singleton_features,1), W_temp_v))
        W_new = np.hstack((W_new, W_temp_h))
        # generate new features z_i row
        Z_new_r = np.hstack((self._Z[[object_index], non_singleton_features], np.ones((len(object_index), K_temp))))
        #print K_temp, object_index, non_singleton_features

        # compute the probability of generating new features / old featrues
        log_old = self.log_likelihood
        Z_new = np.hstack((self._Z[:, non_singleton_features], np.zeros((self._N, K_temp))))
        Z_new[object_index, :] = Z_new_r
        bilinear_matrix = self.bilinear_matrix.copy()
        log_new = self.log_likelihood_Y(Z=Z_new, W=W_new)
        self._overflow = - log_new -1
        prob_new = np.exp(log_new + self._overflow)
        prob_old = np.exp(log_old + self._overflow)

        assert(W_new.shape == (K_new, K_new))
        assert(Z_new.shape == (self._N, K_new))

        # compute the probability of generating new features
        r = prob_new / prob_old
        MH_accept = min(1, r)

        # if we accept the proposal, we will replace old W and Z matrices
        if np.random.random() <= MH_accept and not np.isnan(r):
            # construct W_new and Z_new
            #print 'MH_accept: %s, singleton feature: %s, k_new: %s' % (MH_accept, len(singleton_features), K_temp)
            self._W = W_new
            self._Z = Z_new
            self._K = K_new
            return True
        else:
            self.bilinear_matrix = bilinear_matrix
            self.log_likelihood = log_old
            return False

    """
    Sample W using metropolis hasting """
    def sample_W(self):
        #sys.stdout.write('W')
        # sample every weight
        sigma_rw = 1.0
        if self.symmetric:
            order = np.arange(self._K**2).reshape(self._W.shape)
            iu = np.triu_indices(self._K)
            order = np.random.permutation(order[iu])
        else:
            order = np.random.permutation(self._K**2)
        for (observation_counter, observation_index) in enumerate(order):
            w_old = self._W.flat[observation_index]
            w_new = np.random.normal(w_old, sigma_rw)
            j_new = sp.stats.norm(w_old, sigma_rw).pdf(w_new)
            j_old = sp.stats.norm(w_new, sigma_rw).pdf(w_old)
            if self.assortativity == 2:
                mean = self.z_mean.flat[observation_index]
            else:
                mean = 0
            pw_new = sp.stats.norm(mean, self._sigma_w).pdf(w_new)
            pw_old = sp.stats.norm(mean, self._sigma_w).pdf(w_old)

            log_old = self.log_likelihood
            bilinear_matrix = self.bilinear_matrix.copy()
            self._overflow = - log_old -1
            self._W.flat[observation_index] = w_new
            if self.symmetric:
                indm = np.unravel_index(observation_index, self._W.shape)
                self._W.T[indm] = w_new
            log_new = self.log_likelihood_Y(object_index=observation_index)
            if -log_new < self._overflow:
                self._overflow = - log_new -1
            likelihood_new = np.exp(log_new + self._overflow)
            likelihood_old = np.exp(log_old + self._overflow)
            r = likelihood_new * pw_new * j_old / ( likelihood_old * pw_old * j_new )
            MH_accept = min(1, r)

            if np.random.random() <= MH_accept and not np.isnan(r):
                self.ratio_MH_W += 1
            else:
                self.bilinear_matrix = bilinear_matrix
                self.log_likelihood = log_old
                self._W.flat[observation_index] = w_old
                if self.symmetric:
                    self._W.T[indm] = w_old

        try:
            self.ratio_MH_W /= len(order)
        except:
            pass
        return self.ratio_MH_W

    """
    sample feature variance, i.e., sigma_w """
    def sample_sigma_w(self, sigma_w_hyper_parameter):
        return self.sample_sigma(self._sigma_w_hyper_parameter, self._W - np.tile(self._W_prior, (self._K, 1)))

    """
    remove the empty column in matrix Z and the corresponding feature in W """
    def regularize_matrices(self):
        Z_sum = np.sum(self._Z, axis=0)
        indices = np.nonzero(Z_sum == 0)

        if 0 in Z_sum:
            lgg.warn( "need to regularize matrices, feature to all zeros !")

        #self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]]
        #self._W = self._W[[k for k in range(self._K) if k not in indices], :]
        #self._K = self._Z.shape[1]
        #assert(self._Z.shape == (self._N, self._K))
        #assert(self._W.shape == (self._K, self._K))

    """
    compute the log-likelihood of the data Y """
    def log_likelihood_Y(self, Z=None, W=None, object_index=None):
        if W is None:
            W = self._W
        if Z is None:
            Z = self._Z

        (N, K) = Z.shape
        assert(W.shape == (K, K))

        bilinear_init = self.bilinear_matrix is not None
        #bilinear_init = False

        if type(object_index) is tuple and bilinear_init:
            # Z update
            n = object_index[0]
            k = object_index[1]
            self.bilinear_matrix[n,:] = self.logsigmoid(Z[n].dot(W).dot( Z.T ), self._Y[n,:])
            #self.bilinear_matrix[n,:] = self.logsigmoid( Z.dot(Z[n].dot(W).T).T , self._Y[n,:])
            if self.symmetric:
                self.bilinear_matrix[:,n] = self.bilinear_matrix[n,:]
            else:
                self.bilinear_matrix[:,n] = self.logsigmoid(Z.dot(W).dot(Z[n]), self._Y[:, n])
        elif np.issubdtype(object_index, np.integer) and bilinear_init:
            # W update
            indm = np.unravel_index(object_index, W.shape)
            ki, kj = indm
            sublinear = list( np.where(Z[:, ki] > 0)[0])
            sublinear = sorted(list(set(sublinear + list(np.where(Z[:, kj] > 0)[0]) )))
            if len(sublinear) > 0:
                self.bilinear_matrix[np.ix_(sublinear, sublinear)] = self.logsigmoid( Z[sublinear].dot(W).dot(Z[sublinear].T), self._Y[np.ix_(sublinear, sublinear)] )
                #self.bilinear_matrix[np.ix_(sublinear, sublinear)] = self.logsigmoid( sp_dot(Z[sublinear].dot(W), Z[sublinear].T), Y[np.ix_(sublinear, sublinear)] )
        else:
            # Check speed with sparse matrix here !
            self.bilinear_matrix = self.logsigmoid(Z.dot(W).dot(Z.T))
            #self.bilinear_matrix = self.logsigmoid( Z.dot(Z.dot(W).T).T )

        self.log_likelihood = np.sum(self.bilinear_matrix)
        if np.abs(self.log_likelihood) == np.inf:
            print("dohn debug me here")
            if self._sigb >= 2:
                self._sigb -= 1
            else:
                self._sigb = 1
                self._W /= 2
                W = self._W
            self.bilinear_matrix = self.logsigmoid(Z.dot(W).dot(Z.T))
            self.log_likelihood = np.sum(self.bilinear_matrix)

        return self.log_likelihood

    def logsigmoid(self, X, Y=None):
        if Y is None:
            Y = self._Y
        # 1 - sigmoid(x) = sigmoid(-x) ~ Y * ...
        v = - np.log(1 + np.exp(- Y * self._sigb * X))
        return v

    """
    compute the log-likelihood of W """
    def log_likelihood_W(self):
        log_likelihood = - 0.5 * self._K * self._D * np.log(2 * np.pi * self._sigma_w * self._sigma_w)
        #for k in range(self._K):
        #    W_prior[k, :] = self._mean_a[0, :]
        W_prior = np.tile(self._W_prior, (self._K, 1))
        log_likelihood -= np.trace(np.dot((self._W - W_prior).transpose(), (self._W - W_prior))) * 0.5 / (self._sigma_w ** 2)

        return log_likelihood

    """
    compute the log-likelihood of the model """
    def log_likelihood_model(self):
        # loglikelihood_W ? Z ?
        return self.log_likelihood_Y()

    # @ibp
    def update_hyper(self, hyper):
        if hyper is None:
            return
        elif isinstance(type(hyper), (tuple, list)):
            alpha = hyper[0]
        else:
            alpha = hyper.get('alpha')

        if alpha:
            self._alpha = alpha

    # @ibp
    def get_hyper(self):
        alpha = self._alpha
        delta = (self._mean_w, self._sigma_w)
        return (alpha, delta)

    # @ibp
    def generate(self, N, K=None, nodelist=None, hyperarams=None, mode='predictive', symmetric=True, **kwargs):
        self.update_hyper(hyperarams)
        alpha, delta = self.get_hyper()
        if mode == 'generative':
            N = int(N)

            # Use for the stick breaking generation
            #K = alpha * np.log(N)

            # Generate F
            theta = self.initialize_Z(N, alpha)
            K = theta.shape[1]

            # Generate Phi
            phi = np.random.normal(delta[0], delta[1], size=(K,K))
            if symmetric is True:
                phi = np.triu(phi) + np.triu(phi, 1).T

            self._theta = theta
            self._phi = phi
        elif mode == 'predictive':
            theta, phi = self.get_params()
            K = theta.shape[1]

        if nodelist:
            Y = Y[nodelist, :][:, nodelist]
            phi = phi[nodelist, :]

        likelihood = self.likelihood(theta, phi)
        #likelihood[likelihood >= 0.5 ] = 1
        #likelihood[likelihood < 0.5 ] = 0
        #Y = likelihood
        Y = sp.stats.bernoulli.rvs(likelihood)
        return Y, theta, phi

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            try:
                theta = self._theta
            except:
                theta = self._Z
        if phi is None:
            try:
                phi = self._phi
            except:
                phi = self._W
        bilinear_form = theta.dot(phi).dot(theta.T)
        likelihood = 1 / (1 + np.exp(- self._sigb * bilinear_form))
        return likelihood

    def _reduce_latent(self):
        Z, W = list(map(list, zip(*self.samples)))
        ks = [ mat.shape[1] for mat in Z]
        bn = np.bincount(ks)
        k_win = np.argmax(bn)
        lgg.debug('K selected: %d' % k_win)

        ind_rm = []
        [ind_rm.append(i) for i, v in enumerate(Z) if v.shape[1] != k_win]
        for i in sorted(ind_rm, reverse=True):
            Z.pop(i)
            W.pop(i)

        lgg.debug('Samples Selected: %d over %s' % (len(Z), len(Z)+len(ind_rm) ))

        Z = Z[-1]
        W = np.mean(W, 0)
        self._theta = Z
        self._phi = W
        self._K = self._theta.shape[1]
        return Z, W

    def get_clusters(self, K=None):
        Z, W = self.get_params()
        K = K or Z.shape[1]
        Z = self.leftordered(Z)
        clusters = kmeans(Z, K=K)
        return clusters

    def _get_clusters(self, K=None):
        Z, W = self.get_params()
        K = K or Z.shape[1]
        clusters = np.argmax(Z * np.tile(W.sum(0), (Z.shape[0],1)), 1)
        return clusters

    # add sim optionsin clusters
    def get_communities(self, K=None):
        Z, W = self.get_params()
        K = K or Z.shape[1]
        Z = self.leftordered(Z)
        clusters = kmeans(Z.dot(W), K=K)
        return clusters

    #@wrapper !
    def communities_analysis(self, data, clustering='modularity'):
        symmetric = (data == data.T).all()

        # @debug !!!
        clusters = self.get_clusters()
        #nodes_list = [k[0] for k in sorted(zip(range(len(clusters)), clusters), key=lambda k: k[1])]
        Z, W = self.get_params()
        block_hist = Z.sum(0)

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

        self.comm = {'local_degree':local_degree,
                     'clusters':clusters,
                     'block_hist': block_hist,
                     'size': len(block_hist)}

        return self.comm

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

    def purge(self):
        purge_obj = ['frontend', '_Y']
        for obj in purge_obj:
            if hasattr(self, obj):
                setattr(self, obj, None)


