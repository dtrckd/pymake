# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model.modelbase import GibbsSampler

#@network class frontend :
#   * pb of method conlicts, save, purge etc..fond a way
class immsb_cvb(GibbsSampler):

    # *args ?
    def _init_params(self):
        ### The time Limitations are @heere
        # frontend integration ?
        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = self.frontend.getN()
        _len['nfeat'] = self.frontend.get_nfeat()
        data_ma = self.frontend.data_ma
        _len['nnz'] = self.frontend.ma_nnz()
        _len['nnz_t'] = self.frontend.ma_nnz_t()
        _len['dims'] = self.frontend.ma_dims()
        _len['ones'] = (data_ma == 1).sum()
        _len['zeros'] = (data_ma == 0).sum()
        self._len = _len

        # Stream Parameters
        self.iterations = self.expe.get('iterations', 1)


        # Hyperparams
        delta = self.expe['hyperparams']['delta']
        self.hyper_phi = np.asarray(delta) if isinstance(delta, (np.ndarray, list, tuple)) else np.asarray([delta] * self._len['nfeat'])

        alpha = self.expe['hyperparams']['alpha'] # unused
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        # Sufficient Statistics
        self._ss = self._random_cvb_init()

        self.frontend._set_rawdata_for_likelihood_computation()


    def _random_cvb_init(self):
        K = self._len['K']
        N = self._len['N']
        nfeat = self._len['nfeat']
        dims = self._len['dims']
        zeros = self._len['zeros']
        ones = self._len['ones']
        nnz = self._len['nnz']

        if self._is_symmetric and False:
            self.gamma = np.zeros((N,N,K,K))
            for i, j in self.data_iter():
                gmma = np.random.randint(1, 2*N, (K,K))
                self.frontend.symmetrize(gmma)
                gmma = gmma / gmma.sum()
                self.gamma[i,j] = gmma

                #self.gamma[j,i] = gmma # data_iter 2
                gmma = np.random.randint(1, 2*N, (K,K))
                self.frontend.symmetrize(gmma)
                gmma = gmma / gmma.sum()
                self.gamma[j,i] = gmma
        else:
            self.gamma = np.random.dirichlet(np.ones(K**2)*0.5, (N,N))
            self.gamma.resize(N,N,K,K)
            self.gamma[self.frontend.data_ma == np.ma.masked] = 0 # ???

        #self._symmetric_pt = self._is_symmetric +1
        #self._symmetric_pt = 1

        self.N_theta_left = self.gamma.sum(0).sum(1)
        self.N_theta_right = self.gamma.sum(1).sum(2)

        self.N_phi = np.zeros((nfeat, K, K))
        self.N_phi[0] = self.gamma[self.frontend.data_ma == 0].sum(0)
        self.N_phi[1] = self.gamma[self.frontend.data_ma == 1].sum(0)



    def _reduce_latent(self):
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        self._theta = (theta.T / theta.sum(axis=1)).T

        phi = self.N_phi + np.tile(self.hyper_phi, (self.N_phi.shape[1], self.N_phi.shape[2], 1)).T
        #phi = (phi / np.linalg.norm(phi, axis=0))[1]
        self._phi = (phi / phi.sum(0))[1]

        self._K = self.N_phi.shape[1]

        return self._theta, self._phi

    def _reduce_one(self, i, j):
        xij = self._xij

        self.pik = self.N_theta_left[i] + self.hyper_theta
        self.pjk = self.N_theta_right[j] + self.hyper_theta
        pxk = self.N_phi[xij] + self.hyper_phi[xij]

        ##
        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(pxk) - np.log(self.N_phi.sum(0) + self.hyper_phi_sum)
        ##

        return lognormalize(outer_kk.ravel())

    def sample(self):

        expectation_step = self._len['N']/2

        _it = 0
        for i, j in self.data_iter(randomize=True):

            self._xij = self.frontend.data_ma[i,j]
            self.pull_current(i, j)

            qij = self._reduce_one(i, j).reshape(self._len['K'], self._len['K'])

            self.push_current(i, j, qij)

      #      if _it % expectation_step == 0:
      #          self.N_phi[0] = self.gamma[self.frontend.data_ma == 0].sum(0)
      #          self.N_phi[1] = self.gamma[self.frontend.data_ma == 1].sum(0)

      #      _it += 1

      #  self.N_phi[0] = self.gamma[self.frontend.data_ma == 0].sum(0)
      #  self.N_phi[1] = self.gamma[self.frontend.data_ma == 1].sum(0)


    def pull_current(self, i, j):
        xij = self._xij

        q_left = self.gamma[i,j].sum(0)
        q_right = self.gamma[i,j].sum(1)
        self.N_theta_left[i] -= q_left
        self.N_theta_right[j] -= q_right
        self.N_phi[xij] -= self.gamma[i,j]

        if self._is_symmetric:
            self.N_theta_left[j] -= self.gamma[j,i].sum(0)
            self.N_theta_right[i] -= self.gamma[j,i].sum(1)
            self.N_phi[xij] -= self.gamma[j,i]


    def push_current(self, i, j, qij):
        xij = self._xij

        q_left = qij.sum(0)
        q_right = qij.sum(1)
        self.N_theta_left[i] += q_left
        self.N_theta_right[j] += q_right
        self.N_phi[xij] += qij

        self.gamma[i, j] = qij

        if self._is_symmetric:
            self.N_theta_left[j] += q_left
            self.N_theta_right[i] += q_right
            self.N_phi[xij] += qij

            self.gamma[j, i] = qij

    def compute_entropy(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        pij = self.frontend.data_A * pij + self.frontend.data_B
        ll = np.log(pij).sum()

        # Entropy
        entropy = ll
        #entropy = - ll / self._len['nnz']

        # Perplexity is 2**H(X).

        return entropy

    def compute_entropy_t(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        pij = self.frontend.data_A_t * pij + self.frontend.data_B_t
        ll = np.log(pij).sum()

        # Entropy
        entropy = ll
        #entropy_t = - ll / self._len['nnz_t']

        # Perplexity is 2**H(X).

        return entropy

    def update_hyper(self, hyper):
        pass


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

