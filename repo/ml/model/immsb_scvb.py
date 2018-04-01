import numpy as np
import scipy as sp
from numpy import ma

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model.modelbase import SVB



class immsb_scvb(SVB):

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
        self._len = _len

        self.iterations = self.expe.get('iterations', 1)

        # Chunk Parameters
        self.chunk_size = self._get_chunk()

        self.chunk_len = self._len['nnz'] / self.chunk_size

        if self.chunk_len < 1:
            self.chunk_size = self._len['nnz']
            self.chunk_len = 1

        self._init_gradient()
        #self._init_gradienta()

        # Hyperparams
        delta = self.expe['hyperparams']['delta']
        self.hyper_phi = np.asarray(delta) if isinstance(delta, (np.ndarray, list, tuple)) else np.asarray([delta] * self._len['nfeat'])
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        # Sufficient Statistics
        #self._ss = self._random_s_init()
        self._ss = self._random_ss_init()
        #self._random_cvb_init()

        self.frontend._set_rawdata_for_likelihood_computation()

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

    def _init_gradienta(self):
        self._timestep_a = 0
        self._timestep_b = 0
        self._chi_a = self.expe.get('chi', 1)
        self._tau_a =  self.expe.get('tau', 42)
        self._kappa_a = self.expe.get('kappa', 0.75)
        self._chi_b = 42
        self._tau_b = 300
        self._kappa_b = 0.9
        self._update_gstep_theta()
        self._update_gstep_phi()

    def _init_gradientb(self):
        self._timestep_a = 0
        self._timestep_b = 0
        self._chi_a = 1
        self._tau_a =  42
        self._kappa_a = 0.75
        self._chi_b = self.expe.get('chi', 42)
        self._tau_b = self.expe.get('tau', 300)
        self._kappa_b = self.expe.get('kappa', 0.9)
        self._update_gstep_theta()
        self._update_gstep_phi()

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

        ### Theta recovering
        theta_left = np.zeros((N, K), dtype=float)
        theta_right = np.zeros((N, K), dtype=float)

        self._symmetric_pt = self._is_symmetric +1
        #self._symmetric_pt = 1

        for i, j in self.data_iter(self.frontend.data_ma):
            k_i = z[i, j, 0]
            k_j = z[i, j, 1]
            theta_left[i, k_i] +=  self._symmetric_pt
            theta_right[j, k_j] += self._symmetric_pt

        ##### phi assignment
        phi = np.zeros((nfeat, K, K), dtype=float)

        for i, j in self.data_iter(self.frontend.data_ma):
            z_ij = z[i,j,0]
            z_ji = z[i,j,1]
            phi[self.frontend.data_ma[i, j], z_ij, z_ji] += 1
            if self._is_symmetric:
                phi[self.frontend.data_ma[j, i], z_ji, z_ij] += 1

        self.N_theta_left = theta_left
        self.N_theta_right = theta_right
        self.N_phi = phi

        # Temp Containers (for minibatch)
        self._N_phi = np.zeros((nfeat, K,K))
        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        return [self.N_phi, self.N_theta_right]

    def _random_ss_init(self):
        ''' Sufficient Statistics Initialization '''
        K = self._len['K']
        N = self._len['N']
        nfeat = self._len['nfeat']
        dims = self._len['dims']
        zeros = self._len['zeros']
        ones = self._len['ones']
        nnz = self._len['nnz']

        #N_theta_left = (dims[:, None] * np.random.dirichlet(np.ones(K), N))
        #N_theta_right = (dims[:, None] * np.random.dirichlet(np.ones(K), N))
        N_theta_left = (dims[:, None] * np.random.dirichlet([0.5]*K, N))
        N_theta_right = (dims[:, None] * np.random.dirichlet([0.5]*K, N))

        #N_phi = np.random.dirichlet([zeros, ones], K**2).T.reshape(2,K,K)
        #N_phi = nnz / (K**2) *  np.random.dirichlet([1, 1], K**2).T.reshape(2,K,K)

        N_phi = np.zeros((2,K,K))
        N_phi[0] = np.random.dirichlet([0.5]*K**2).reshape(K,K) * zeros
        N_phi[1] = np.random.dirichlet([0.5]*K**2).reshape(K,K) * ones

        self.N_phi = N_phi
        self.N_theta_left = N_theta_left
        self.N_theta_right = N_theta_right


        # Temp Containers (for minibatch)
        self._N_phi = np.zeros((nfeat, K,K))
        self.hyper_phi_sum = self.hyper_phi.sum()
        self.hyper_theta_sum = self.hyper_theta.sum()

        #self._qij = self.likelihood(*self._reduce_latent())
        self._symmetric_pt = self._is_symmetric +1

        # Return sufficient statistics
        return [N_phi, N_theta_left, N_theta_right]

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
                gmma = np.random.randint(1, N, (K,K))
                self.frontend.symmetrize(gmma)
                gmma = gmma / gmma.sum()
                self.gamma[i,j] = gmma

                #self.gamma[j,i] = gmma # data_iter 2
                gmma = np.random.randint(1, N, (K,K))
                self.frontend.symmetrize(gmma)
                gmma = gmma / gmma.sum()
                self.gamma[j,i] = gmma
        else:
            self.gamma = np.random.dirichlet(np.ones(K**2)*0.5, (N,N))
            self.gamma.resize(N,N,K,K)
            #self.gamma[self.frontend.data_ma == np.ma.masked] = 0 # ???

        #self._symmetric_pt = 1

        self.N_theta_left = self.gamma.sum(0).sum(1)
        self.N_theta_right = self.gamma.sum(1).sum(2)

        self.N_phi = np.zeros((nfeat, K, K))
        self.N_phi[0] = self.gamma[self.frontend.data_ma == 0].sum(0)
        self.N_phi[1] = self.gamma[self.frontend.data_ma == 1].sum(0)


    def _reset_containers(self):
        self._N_phi *= 0
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

        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(pxk) - np.log(self.N_phi.sum(0) + self.hyper_phi_sum)

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
        self._N_phi[xij] += qij * self._symmetric_pt


    def _purge_minibatch(self):
        ''' Update the global gradient then purge containers '''
        if not self._is_container_empty():
            self.N_phi = (1 - self.gstep_phi)*self.N_phi + self.gstep_phi * (self._len['nnz'] / len(self.samples)) * self._N_phi

        self._update_gstep_phi()

        self._reset_containers()

        self._timestep_b += 1


    def compute_entropy(self):
        pij = self.likelihood(*self._reduce_latent())

        # Log-likelihood
        pij = self.frontend.data_A * pij + self.frontend.data_B
        #ll = np.log(pij).sum()
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
        entropy_t = ll

        #entropy_t = - ll / self._len['nnz_t']
        # Perplexity is 2**H(X).

        return entropy_t

    def compute_elbo(self):

        pij = self.likelihood(*self._reduce_latent())

        pij = ma.array(pij, mask=self.get_mask())

        # Log-likelihood
        ll = self.frontend.data_A * pij + self.frontend.data_B


        eq = (pij * np.log(ll)).sum()
        hq = (pij * np.log(pij)).sum()

        elbo = eq - hq

        return elbo


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

