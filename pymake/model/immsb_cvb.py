# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from pymake.model.modelbase import GibbsSampler

#@network class frontend :
#   * pb of method conlicts, save, purge etc..fond a way
class immsb_cvb(GibbsSampler):

    # *args ?
    def _init_params(self):
        ### The time Limitations are @heere
        # frontend integration ?
        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = self.fr.getN()
        _len['nfeat'] = self.fr.get_nfeat()
        data_ma = self.fr.data_ma
        _len['nnz'] = self.fr.ma_nnz()
        _len['dims'] = self.fr.ma_dims()
        _len['ones'] = (data_ma == 1).sum()
        _len['zeros'] = (data_ma == 0).sum()
        self._len = _len

        # Stream Parameters
        self.iterations = self.expe.get('iterations', 1)


        # Hyperparams
        delta = self.expe['hyperparams']['delta']
        self.hyper_phi = np.asarray(delta) if isinstance(delta, (np.ndarray, list, tuple)) else np.asarray([delta] * self._len['nfeat'])
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        # Sufficient Statistics
        self._ss = self._random_s_init()
        #self._ss = self._random_ss_init()

        # JUNK
        # for loglikelihood bernoulli computation
        data_ma = self.fr.data_ma
        self.data_A = data_ma.copy()
        self.data_A.data[self.data_A.data == 0] = -1
        self.data_B = np.ones(data_ma.shape) - data_ma


    def _random_s_init(self):
        K = self._len['K']
        N = self._len['N']
        nfeat = self._len['nfeat']
        dims = self._len['dims']
        zeros = self._len['zeros']
        ones = self._len['ones']
        nnz = self._len['nnz']

        ##### Topic assignment
        dim = (N, N, 2)

        # Poisson way
        #alpha_0 = self.alpha_0
        #z = np.array( [poisson(alpha_0, size=dim) for dim in data_dims] )

        # Random way
        K = self._len['K']
        z = np.random.randint(0, K, (dim))

        if self._is_symmetric:
            z[:, :, 0] = np.triu(z[:, :, 0]) + np.triu(z[:, :, 0], 1).T
            z[:, :, 1] = np.triu(z[:, :, 1]) + np.triu(z[:, :, 1], 1).T


        self.gamma = np.zeros(N, N, K, K,  dtype=float)
        phi = np.zeros((nfeat, K, K), dtype=float)
        theta_right = np.zeros((N, K), dtype=float)
        theta_left = np.zeros((N, K), dtype=float)
        self.symmetric_pt = self._is_symmetric +1

        for i, j in self.data_iter(self.fr.data_ma):
            k_i = z[i, j, 0]
            k_j = z[i, j, 1]
            self.gamma[i, j, k_i, k_j] += self.symmetric_pt
            theta_left[i, k_i] +=  self.symmetric_pt
            theta_right[j, k_j] += self.symmetric_pt
            phi[self.fr.data_ma[i, j], z_ij, z_ji] += 1
            if self._is_symmetric:
                phi[self.fr.data_ma[j, i], z_ji, z_ij] += 1

        self.N_theta_right = theta_right
        self.N_theta_left = theta_left
        self.N_phi = phi


        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        return [self.N_phi, self.N_theta_right]



    def update_hyper(self, hyper):
        pass


    def data_iter(self, data, randomize=False):
        data_ma = data

        order = np.arange(data_ma.size).reshape(data_ma.shape)
        masked = order[data_ma.mask]

        if self._is_symmetric:
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


    def entropy(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        pij = self.data_A * pij + self.data_B
        lpij = np.log(pij)

        ll = lpij.sum()

        # Entropy
        self._entropy = - ll / self._len['nnz']

        # Perplexity is 2**H(X).

        return self._entropy

    def _reduce_latent(self):
        #theta = self.N_theta_right + np.tile(self.hyper_theta, (self.N_theta_right.shape[0],1))
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        theta = (theta.T / theta.sum(axis=1)).T

        phi = self.N_phi + np.tile(self.hyper_phi, (self.N_phi.shape[1], self.N_phi.shape[2], 1)).T
        #phi = (phi / np.linalg.norm(phi, axis=0))[1]
        phi = (phi / phi.sum(0))[1]

        self._theta = theta
        self._phi = phi
        self._K = self.N_phi.shape[1]

        return theta, phi

    def _reduce_one(self, i, j):
        xij = self._xij

        self.pik = self.N_theta_right[i] + self.hyper_theta
        self.pjk = self.N_theta_left[j] + self.hyper_theta
        #self.pjk = self.N_theta_right[j] + self.hyper_theta
        pxk = self.N_phi[xij] + self.hyper_phi[xij]

        ##
        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(pxk) - np.log(self.N_phi.sum(0) + self.hyper_phi_sum)
        ##

        if self._is_symmetric:
            self.fr.symmetrize(outer_kk)

        return lognormalize(outer_kk)

    def sample()

        for i, j in self.data_iter(randomize=True):

            xij = self.fr.data_ma[i,j]
            self.pull_current(i, j, xij)

            qij = self._reduce_one(i, j)

            self.push_curent(i, j, qij)

    def pull_current(i, j, xij):
        self.N_theta_right[i] -= self.gamma[i,j].sum(0)
        self.N_theta_right[j] -= self.gamma[i,j].sum(1)
        self.N_phi[xij] -= self.gamma[i,j]

    def push_current(i, j, qij):
        self.N_theta_right[i] += qij.sum(0)
        self.N_theta_right[j] += qij.sum(1)
        self.N_phi[xij] += qij

        self.gamma[i, j] = qij


    def generate(self, N=None, K=None, hyperparams=None, mode='predictive', symmetric=True, **kwargs):
        #self.update_hyper(hyperparams)
        #alpha, gmma, delta = self.get_hyper()

        # predictive
        try: theta, phi = self.get_params()
        except: return self.generate(N, K, hyperparams, 'generative', symmetric)
        K = theta.shape[1]

        pij = self.likelihood(theta, phi)
        pij = np.clip(pij, 0, 1)
        Y = sp.stats.bernoulli.rvs(pij)

        return Y


if __name__ == "__main__":

    import pymake
    from pymake import frontendNetwork

    data = np.arange(16).reshape(4,4)
    data_ma = np.ma.array(data, mask = data*0)

    data_ma.mask[0,1] = True
    data_ma.mask[1,1] = True
    data_ma.mask[1,2] = True
    data_ma.mask[1,3] = True
    data_ma.mask[3,3] = True

    fr = frontendNetwork.from_array(data_ma)

    model = immsb_scvb({}, fr)

    _abc = model.data_iter(data_ma)

    # data to iter
    print(data_ma[zip(*_abc)])

