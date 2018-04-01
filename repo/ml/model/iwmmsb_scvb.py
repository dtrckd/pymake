import numpy as np
import scipy as sp
import scipy.stats
from numpy import ma

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model.modelbase import SVB

class iwmmsb_scvb(SVB):

    # *args ?
    def _init_params(self):
        self.limit_elbo_diff = 1e-3
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
        _len['nnzsum'] = self.frontend.data_ma.compressed().sum()
        self._len = _len

        self.iterations = self.expe.get('iterations', 1)

        # Chunk Parameters
        self.chunk_size = self._get_chunk()

        self.chunk_len = self._len['nnz'] / self.chunk_size

        if self.chunk_len < 1:
            self.chunk_size = self._len['nnz']
            self.chunk_len = 1

        self._init_gradient()

        # Hyperparams
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        self.hyper_phi = np.asarray(self.expe['delta'])

        # Sufficient Statistics
        self._ss = self._random_ss_init()

        #self.frontend._set_rawdata_for_likelihood_computation()

    def _init_gradient(self):
        self._timestep_a = 0
        self._timestep_b = 0
        self._chi_a = self.expe.get('chi_a', 5)
        self._tau_a = self.expe.get('tau_a', 10)
        self._kappa_a = self.expe.get('kappa_a', 0.9)
        self._chi_b = self.expe.get('chi_b', 1)
        self._tau_b = self.expe.get('tau_b', 100)
        self._kappa_b = self.expe.get('kappa_b', 0.9)
        self._update_gstep_theta()
        self._update_gstep_phi()


    def _random_ss_init(self):
        ''' Sufficient Statistics Initialization '''
        K = self._len['K']
        N = self._len['N']
        nfeat = self._len['nfeat']
        dims = self._len['dims']
        zeros = self._len['zeros']
        ones = self._len['ones']
        nnz = self._len['nnz']
        nnzsum = self._len['nnzsum']

        #N_theta_left = (dims[:, None] * np.random.dirichlet(np.ones(K), N))
        #N_theta_right = (dims[:, None] * np.random.dirichlet(np.ones(K), N))
        N_theta_left = (dims[:, None] * np.random.dirichlet([0.5]*K, N))
        N_theta_right = (dims[:, None] * np.random.dirichlet([0.5]*K, N))

        N_phi = np.random.dirichlet([0.5]*K**2).reshape(K,K) *nnz

        self.N_phi = N_phi
        self.N_theta_left = N_theta_left
        self.N_theta_right = N_theta_right


        # Temp Containers (for minibatch)
        self._N_phi = np.zeros((K,K))
        self._N_Y = np.zeros((K,K))
        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        #self.N_Y = np.random.poisson(0.1, (K,K)) * N
        self.N_Y = np.random.dirichlet([0.1]*K**2).reshape(K,K) * nnzsum

        #self._qij = self.likelihood(*self._reduce_latent())
        self._symmetric_pt = self._is_symmetric +1

        # Return sufficient statistics
        return [N_phi, N_theta_left, N_theta_right]


    def _reset_containers(self):
        self._N_phi *= 0
        self._N_Y *= 0
        self.samples = []
        return

    def _is_container_empty(self):
        return len(self.samples) == 0

    def _update_gstep_theta(self):
        ''' Gradient converge for kappa _in (0.5,1] '''
        chi = self._chi_a
        tau = self._tau_a
        kappa = self._kappa_a

        self.gstep_theta = chi / ((tau + self._timestep_a)**kappa)

    def _update_gstep_phi(self):
        chi = self._chi_b
        tau = self._tau_b
        kappa = self._kappa_b

        self.gstep_phi =  chi / ((tau + self._timestep_b)**kappa)


    def _reduce_latent(self):
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        self._theta = (theta.T / theta.sum(axis=1)).T

        k = self.N_Y + self.hyper_phi[0]
        p = (self.N_phi + self.hyper_phi[1] + 1)**-1
        self._phi = lambda x:sp.stats.nbinom.pmf(x, k, 1-p)
        #mean = k*p / (1-p)
        #var = k*p / (1-p)**2

        self._K = self.N_phi.shape[1]

        return self._theta, self._phi

    def get_nb_ss(self):
        k = self.N_Y + self.hyper_phi[0]
        p = (self.N_phi + self.hyper_phi[1] + 1)**-1
        mean = k*p / (1-p)
        var = k*p / (1-p)**2
        return mean, var


    def _reduce_one(self, i, j):
        xij = self._xij

        self.pik = self.N_theta_left[i] + self.hyper_theta
        self.pjk = self.N_theta_right[j] + self.hyper_theta

        k = self.N_Y + self.hyper_phi[0]
        p = (self.N_phi + self.hyper_phi[1] + 1)**-1

        kernel = sp.stats.nbinom.pmf(xij, k, 1-p)
        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(kernel)

        return lognormalize(outer_kk.ravel())


    def maximization(self, iter):
        ''' Variational Objective '''
        i,j = iter
        #self.pull_current(i, j)
        variational = self._reduce_one(i,j).reshape(self._len['K'], self._len['K'])
        #self.push_current(i, j, variational)
        self.samples.append(variational)

    def expectation(self, iter, burnin=False):
        ''' Follow the White Rabbit '''
        i,j = iter
        xij = self._xij
        qij = self.samples[-1]

        self._update_local_gradient(i, j, qij)
        self._timestep_a += 1

        if burnin:
            self.samples = []
            return
        else:
            self._update_global_gradient(i, j, qij, xij)
            #self._purge_minibatch()
            #self._timestep_b += 1
            pass


    def _update_local_gradient(self, i, j, qij):
        _len = self._len

        # Sum ?
        q_left = qij.sum(0)
        q_right = qij.sum(1)

        self.N_theta_left[i] = (1 - self.gstep_theta)*self.N_theta_left[i] + (self.gstep_theta * _len['dims'][i] * q_left)
        self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta * _len['dims'][j] * q_right)

        if self._is_symmetric:
            self.N_theta_left[j] = (1 - self.gstep_theta)*self.N_theta_left[j] + (self.gstep_theta * _len['dims'][j] * q_left)
            self.N_theta_right[i] = (1 - self.gstep_theta)*self.N_theta_right[i] + (self.gstep_theta * _len['dims'][i] * q_right)


        # or Mean ?
        #self.N_theta_left[i]  = (1 - self.gstep_theta)*self.N_theta_left[i]  + (self.gstep_theta * _len['dims'][i] * qij.mean(1))
        #self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta * _len['dims'][j] * qij.mean(0))

        # or Right  ?
        #pik = self.pik / (2*len(_len['dims']) + self.hyper_theta_sum)
        #pjk = self.pjk / (2*len(_len['dims']) + self.hyper_theta_sum)
        #self.N_theta_left[i]  = (1 - self.gstep_theta)*self.N_theta_left[i]  + (self.gstep_theta * _len['dims'][i] * pik)
        #self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta * _len['dims'][j] * pjk)

        self._update_gstep_theta()

    def _update_global_gradient(self, i, j, qij, xij):
        self._N_phi += qij * self._symmetric_pt
        self._N_Y += xij * qij * self._symmetric_pt


    def _purge_minibatch(self):
        ''' Update the global gradient then purge containers '''
        if not self._is_container_empty():
            self.N_phi = (1 - self.gstep_phi)*self.N_phi + self.gstep_phi * (self._len['nnz'] / len(self.samples)) * self._N_phi

            self.N_Y = (1 - self.gstep_phi)*self.N_Y + self.gstep_phi * (self._len['nnz'] / len(self.samples)) * self._N_Y


        self._update_gstep_phi()

        self._reset_containers()

        self._timestep_b += 1

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        data = self.frontend.data_ma
        likelihood = ma.array(np.zeros(data.shape))
        for i,j in self.data_iter():
            l = theta[i].dot(phi(data[i,j])).dot(theta[j])
            likelihood[i,j] = l or ma.masked

        #likelihood = theta.dot(kern).dot(theta.T)
        return likelihood

    def compute_entropy(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        ll = np.log(pij).sum()

        # Entropy
        entropy = ll
        #self._entropy = - ll / self._len['nnz']

        # Perplexity is 2**H(X).

        return entropy

    def compute_entropy_t(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        ll = np.log(pij).sum()

        # Entropy
        entropy_t = ll
        #self._entropy_t = - ll / self._len['nnz_t']

        # Perplexity is 2**H(X).

        return entropy_t

    def compute_elbo(self):
        # how to compute elbo for all possible links weights, mean?
        return None

    def compute_roc(self):
        return None


    def update_hyper(self, hyper):
        pass

    def generate(self, N=None, K=None, hyperparams=None, mode='predictive', symmetric=True, **kwargs):
        #self.update_hyper(hyperparams)
        #alpha, gmma, delta = self.get_hyper()

        # predictive
        try: theta, phi = self.get_params()
        except: return self.generate(N, K, hyperparams, 'generative', symmetric)
        K = theta.shape[1]

        raise NotImplementedError

        pij = self.likelihood(theta, phi)
        pij = np.clip(pij, 0, 1)
        Y = sp.stats.bernoulli.rvs(pij)

        return Y



