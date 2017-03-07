# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from .modelbase import SVB

#@network class frontend :
#   * pb of method conlicts, save, purge etc..fond a way
class immsb_scvb(SVB):

    # @Debug (graph selfloop in initialization ? ma ?)
    csv_typo = ''

    # *args ?
    def _init_params(self, frontend):
        ### The time Limitations are @heere
        # frontend integration ?
        self._timestep = 0
        _len = {}
        _len['nnz'] = frontend.ma_nnz()
        _len['nfeat'] = frontend.get_nfeat()
        _len['K'] = self.expe.get('K')
        _len['N'] = frontend.getN()
        _len['dims'] = frontend.ma_dims()
        data_ma = frontend.data_ma
        _len['ones'] = frontend.data_ma[data_ma == 1].sum()
        _len['zeros'] = frontend.data_ma[data_ma == 0].sum()
        self._len = _len

        # Stream Parameters
        chunk = self.expe.get('chunk', 10)
        self.chunk_size = chunk * self._len['N']
        self.chunk_len = self._len['nnz']/self.chunk_size

        if self.chunk_len < 1:
            self.chunk_size = self._len['nnz']
            self.chunk_len = 1

        self.gradient_update_freq = self.chunk_size / 100
        #self.burnin = 1
        self.burnin = 10
        self._update_gstep_theta()
        self._update_gstep_phi()

        # Hyperparams
        delta = self.expe['hyperparams']['delta']
        self.hyper_phi = np.asarray(delta) if isinstance(delta, (np.ndarray, list, tuple)) else np.asarray([delta] * self._len['nfeat'])
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        # Sufficient Statistics
        self._ss = self._random_ss_init(frontend)

        # JUNK
        # for loglikelihood bernoulli computation
        data_ma = self.fr.data_ma
        self.data_A = data_ma.copy()
        self.data_A.data[self.data_A.data == 0] = -1
        self.data_B = np.ones(data_ma.shape) - data_ma

        self.elbo = self.perplexity()

    def _update_gstep_phi(self, kappa=0.5):
        self.gstep_phi =  (1 +self._timestep)**-kappa

    def _update_gstep_theta(self, kappa=0.5):
        self.gstep_theta =  (1 +self._timestep)**-kappa

    def _random_ss_init(self, frontend):
        ''' Sufficient Statistics Initialization '''
        K = self._len['K']
        N = self._len['N']
        nfeat = self._len['nfeat']
        zeros = self._len['zeros']
        ones = self._len['ones']

        N_theta_left = (2*N * np.random.dirichlet(np.ones(K), N)).astype(int)
        N_theta_right = (2*N * np.random.dirichlet(np.ones(K), N)).astype(int)

        N_phi = np.zeros((nfeat, K,K))
        N_phi[0] = (zeros * np.random.dirichlet(np.ones(K*K))).astype(int).reshape(K,K)
        N_phi[1] = (ones  * np.random.dirichlet(np.ones(K*K))).astype(int).reshape(K,K)

        self.N_phi = N_phi
        self.N_theta_left = N_theta_left
        self.N_theta_right = N_theta_right

        # Temp Containers (for minibatch)
        self._N_phi = np.zeros((nfeat, K,K), dtype=int)
        self._N_phi_sum = self.N_phi.sum(0)
        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        return [N_phi, N_theta_left, N_theta_right]

    def _reset_containers(self):
        self._N_phi *= 0
        self.samples = []
        return

    def _is_container_empty(self):
        return len(self.samples) == 0

    def update_hyper(self, hyper):
        pass

    def data_iter(self, batch, randomize=True):
        data_ma = batch

        order = np.arange(data_ma.size).reshape(data_ma.shape)
        masked = order[data_ma.mask]

        if self.fr.is_symmetric():
            tril = np.tril_indices_from(data_ma, -1)
            tril = order[tril]
            masked =  np.append(masked, tril)

        # Remove masked value to the iteration list
        order = np.delete(order, masked)
        # Get the indexes of nodes (i,j) for each observed interactions
        order = list(zip(*np.unravel_index(order, data_ma.shape)))

        if randomize is True:
            np.random.shuffle(order)
        return order

    def get_elbo(self):
        return elbo

    def perplexity(self):
        pij = self.likelihood(*self._reduce_latent())

        p_ij = self.data_A * pij + self.data_B
        pp = np.log(pij).sum()
        pp = - pp / self._len['nnz']
        return pp

    def _reduce_latent(self):
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        theta = (theta.T / theta.sum(axis=1)).T

        phi = self.N_phi + np.tile(self.hyper_phi, (self.N_phi.shape[1], self.N_phi.shape[2], 1)).T
        phi = (phi / np.linalg.norm(phi, axis=0))[1]

        return theta, phi

    def _reduce_one(self, i, j):
        xij = self._xij
        self.pik = self.N_theta_left[i] + self.hyper_theta
        self.pjk = self.N_theta_right[j] + self.hyper_theta
        pxk = self.N_phi[xij] + self.hyper_phi[xij]
        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(pxk) - np.log(self._N_phi_sum + self.hyper_phi_sum)
        return lognormalize(outer_kk)

    def maximization(self, iter):
        ''' Variational Objective '''
        i,j = iter
        variational = self._reduce_one(i,j)
        self.samples.append(variational)

    def expectation(self, iter):
        ''' Follow the White Rabbit '''
        i,j = iter
        xij = self._xij
        qij = self.samples[-1]

        self._update_local_gradient(i, j, qij)
        if self._id_burn < self.burnin-1:
            return
        self._update_global_gradient(i, j, qij, xij)

    def _update_local_gradient(self, i, j, qij):
        _len = self._len

        #self.N_theta_left[i] = (1 - self.gstep_theta)*self.N_theta_left[i] + (self.gstep_theta* _len['dims'][i] * qij.sum(1)).astype(int)
        #self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta* _len['dims'][j] * qij.sum(0)).astype(int)

        pik = self.pik / (len(2*_len['dims']) + self.hyper_theta_sum)
        pjk = self.pjk / (len(2*_len['dims']) + self.hyper_theta_sum)
        self.N_theta_left[i] = (1 - self.gstep_theta)*self.N_theta_left[i] + (self.gstep_theta* _len['dims'][i] * pik).astype(int)
        self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta* _len['dims'][j] * pjk).astype(int)

    def _update_global_gradient(self, i, j, qij, xij):
        _len = self._len

        if self.gradient_update_freq <= 1:
            self.N_phi[xij] = (1 - self.gstep_phi)*self.N_phi[xij] + self.gstep_phi* (_len['nnz'] * qij).astype(int)
            self._N_phi_sum = self.N_phi.sum(0)
        else:
            self._N_phi[xij] += (_len['nnz'] * qij).astype(int)
            if self._id_token % self.gradient_update_freq == 0:
                self._purge_minibatch()
            self._timestep += 1
            self._update_gstep_theta()
            self._update_gstep_phi()

    def _purge_minibatch(self):
        ''' Update the global gradient then purge containers'''
        if not self._is_container_empty():
            self.N_phi = (1 - self.gstep_phi)*self.N_phi + self.gstep_phi*self._N_phi / self.gradient_update_freq
            self._N_phi_sum = self.N_phi.sum(0)
            self._reset_containers()
        else:
            pass
