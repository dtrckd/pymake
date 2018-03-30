from time import time
import sys
import numpy as np
import scipy as sp
from numpy import ma
import scipy.stats

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model.modelbase import SVB

#import warnings
#warnings.filterwarnings('error')
#warnings.catch_warnings()
##np.seterr(all='print')


class iwmmsb_scvb2(SVB):

    _purge = ['_kernel', '_lut_nbinom']

    def _init_params(self, frontend):
        self.frontend = frontend

        # Save the testdata
        data_test = np.transpose(self.frontend.data_test.nonzero())
        frontend.reverse_filter()
        weights = []
        for i,j in data_test:
            weights.append(frontend.weight(i,j))
        frontend.reverse_filter()
        self._data_test = np.hstack((data_test, np.array(weights)[None].T))

        # Data statistics
        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = frontend.num_nodes()
        _len['nnz'] = frontend.num_nnz()
        #_len['nnz_t'] = frontend.num_nnz_t()
        _len['dims'] = frontend.num_neighbors()
        _len['nnzsum'] = frontend.num_nnzsum()
        self._len = _len

        self._K = self._len['K']
        self._is_symmetric = frontend.is_symmetric()
        self.limit_elbo_diff = 1e-3

        self._init_gradient()

        # Hyperparams
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        self.hyper_phi = np.asarray(self.expe['delta'])

        self._random_ss_init()

    def _init_gradient(self):
        self._timestep_a = 0
        self._timestep_b = 0
        self._timestep_c = 0
        self._chi_a = self.expe.get('chi_a', 5)
        self._tau_a = self.expe.get('tau_a', 10)
        self._kappa_a = self.expe.get('kappa_a', 0.9)
        self._chi_b = self.expe.get('chi_b', 1)
        self._tau_b = self.expe.get('tau_b', 100)
        self._kappa_b = self.expe.get('kappa_b', 0.9)
        self._update_gstep_theta()
        self._update_gstep_phi()
        self._update_gstep_y()


    def _random_ss_init(self):
        ''' Sufficient Statistics Initialization '''
        K = self._len['K']
        N = self._len['N']
        nnz = self._len['nnz']
        nnzsum = self._len['nnzsum']
        dims = self._len['dims']

        self.N_theta_left = (dims[:, None] * np.random.dirichlet([0.5]*K, N))
        self.N_theta_right = (dims[:, None] * np.random.dirichlet([0.5]*K, N))

        self.N_phi = np.random.dirichlet([0.5]*K**2).reshape(K,K) *nnz

        #self.N_Y = np.random.poisson(0.1, (K,K)) * N
        self.N_Y = np.random.dirichlet([0.1]*K**2).reshape(K,K) * nnzsum

        if self._is_symmetric:
            self.N_theta_left = self.N_theta_right
            self.N_phi = np.triu(self.N_phi) + np.triu(self.N_phi, 1).T
            self.N_Y = np.triu(self.N_Y) + np.triu(self.N_Y, 1).T


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

    def _update_gstep_y(self):
        chi = self._chi_b
        tau = self._tau_b
        kappa = self._kappa_b

        self.gstep_y =  chi / ((tau + self._timestep_c)**kappa)


    def _reduce_latent(self):
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        self._theta = (theta.T / theta.sum(axis=1)).T

        k = self.N_Y + self.hyper_phi[0]
        p = (self.N_phi + self.hyper_phi[1] + 1)**-1
        self._phi = lambda x:sp.stats.nbinom.pmf(x, k, 1-p)
        #mean = k*p / (1-p)
        #var = k*p / (1-p)**2

        return self._theta, self._phi

    def _reduce_one(self, i, j, xij, update_kernel=True):

        if self._is_symmetric:
            self.pik = self.pjk = self.N_theta_left[i] + self.hyper_theta
            self.pjk = self.pik
        else:
            self.pik = self.N_theta_left[i] + self.hyper_theta
            self.pjk = self.N_theta_right[j] + self.hyper_theta

        if update_kernel:
            k = self.N_Y + self.hyper_phi[0]
            p = (self.N_phi + self.hyper_phi[1] + 1)**-1
            # @debug: Some invalie values here sometime !!
            self._kernel = lambda x:sp.stats.nbinom.pmf(x, k, 1-p)
            self._lut_nbinom = [sp.stats.nbinom.pmf(x, k, 1-p) for x in range(42)]
            #kernel = sp.stats.nbinom.pmf(xij, k, 1-p)

        if len(self._lut_nbinom) > xij:
            # Wins some times...
            kernel = self._lut_nbinom[xij]
        else:
            kernel = self._kernel(xij)

        # debug: Underflow
        kernel[kernel<=1e-300] = 1e-300

        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(kernel)

        return lognormalize(outer_kk.ravel())

    def get_nb_ss(self):
        k = self.N_Y + self.hyper_phi[0]
        p = (self.N_phi + self.hyper_phi[1] + 1)**-1
        mean = k*p / (1-p)
        var = k*p / (1-p)**2
        return mean, var

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        lut_nbinom = [phi(x) for x in range(42)]

        likelihood = []

        for i,j, xij in self._data_test:

            if len(lut_nbinom) > xij:
                # Wins some times...
                kernel = lut_nbinom[xij]
            else:
                kernel = phi(xij)
            l = theta[i].dot(kernel).dot(theta[j])
            likelihood.append(l or ma.masked)

        likelihood = ma.array(likelihood)
        #likelihood = theta.dot(kern).dot(theta.T)

        return likelihood

    def compute_entropy(self):
        return self.compute_entropy_t()

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

    def compute_measures(self, begin_it=0):
        ''' Compute measure as model attributes.
            begin_it: is the time of the begining of the iteration.
        '''

        if self.expe.get('deactivate_measures'):
            return

        for meas in self.expe._csv_typo.split():

            if meas == '_entropy':
                old_entropy = self._entropy
                _meas = self.compute_entropy()
                self.entropy_diff = _meas - old_entropy
            elif hasattr(self, 'compute'+meas):
                _meas = getattr(self, 'compute'+meas)()
            else:
                _meas = getattr(self, meas)

            setattr(self, meas, _meas)


        self.time_it = (time() - begin_it)


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

    def fit(self, frontend):
        ''' chunk is the number of row to threat in a minibach '''

        self._init(frontend)
        mnb_total = frontend.num_mnb()

        # Init sampling variables
        observed_pt = 0
        mnb_num = 0
        vertex = None

        self._entropy = self.compute_entropy()
        print( '__init__ Entropy: %f' % self._entropy)
        for _it, obj in enumerate(frontend):

            source, target, weight = obj
            if type(source) is str:
                #print(_it, source, target, weight)
                _set_pos = source
                _vertex = target['vertex']
                _direction = target['direction']
                _scaler = weight
                _qij_samples = []
                _node_idxs = []
                _weights = []
                new_mnb = True

                update_kernel = True
            else:
                i = source
                j = target
                weights.append(weight)
                if direction == 0:
                    node_idxs.append(j)
                else:
                    node_idxs.append(i)

                # Maximization
                qij_samples.append( self._reduce_one(i,j, weight, update_kernel).reshape(self._len['K'], self._len['K']) )

                # Update local gradient / Expectation
                # will be longer but faster ???
                # try updating only  J => ima sure it will increase perf (j here and i in the global set (ore reversly selon la direction)
                #
                #self.N_theta_left[i] = (1 - self.gstep_theta)*self.N_theta_left[i] + (self.gstep_theta * scaler * qij_samples[-1].sum(0))
                #self.N_theta_right[j] = (1 - self.gstep_theta)*self.N_theta_right[j] + (self.gstep_theta * scaler * qij_samples[-1].sum(1))
                #self._timestep_a += 1
                #self._update_gstep_theta()
                update_kernel = False

                observed_pt += 1

            if vertex is None or not qij_samples:
                # Enter here only once !%!
                set_pos = _set_pos
                vertex = _vertex
                direction = _direction
                scaler = _scaler
                qij_samples = _qij_samples
                node_idxs = _node_idxs
                weights = _weights
                new_mnb = False
                continue

            if new_mnb:

                qijs = np.asarray(qij_samples)
                # Update global gradient / Expectation
                # Normalize or not ?
                #norm=1
                norm = qijs.shape[0]
                qijs_sum = qijs.sum(0)

                if direction == 0:
                    self.N_theta_left[i] = (1-self.gstep_theta)*self.N_theta_left[i] + self.gstep_theta*scaler*qijs_sum.sum(0) /norm
                    self.N_theta_right[node_idxs] = (1-self.gstep_theta)*self.N_theta_right[node_idxs] + self.gstep_theta*scaler*qijs.sum(2) /norm
                else:
                    self.N_theta_left[node_idxs] = (1-self.gstep_theta)*self.N_theta_left[node_idxs] + self.gstep_theta*scaler*qijs.sum(1) /norm
                    self.N_theta_right[j] = (1-self.gstep_theta)*self.N_theta_right[j] + self.gstep_theta*scaler*qijs_sum.sum(1) /norm

                self._timestep_a += scaler
                self._update_gstep_theta()

                #scaler = self._len['nnz']
                self.N_phi = (1 - self.gstep_phi)*self.N_phi + self.gstep_phi * scaler * qijs_sum /norm
                if set_pos != '0':
                    self.N_Y = (1 - self.gstep_y)*self.N_Y + self.gstep_y * scaler * np.sum([weights[n]*qijs[n] for n in range(len(weights)) ],0) /norm
                    self._timestep_c += scaler
                    self._update_gstep_y()

                self._timestep_b += scaler
                self._update_gstep_phi()

                # Allocate current state variable
                set_pos = _set_pos
                vertex = _vertex
                direction = _direction
                scaler = _scaler
                qij_samples = _qij_samples
                node_idxs = _node_idxs
                weights = _weights
                new_mnb = False

                mnb_num += 1

                if mnb_num % 50 == 0:
                    prop_edge = observed_pt / self._len['nnz']
                    self._observed_pt = observed_pt
                    self.compute_measures()

                    print('.', end='')
                    self.log.info('it %d | prop edge: %.2f | mnb %d/%d, %s, Entropy: %f,  diff: %f' % (_it, prop_edge,
                                                                                                       mnb_num, mnb_total,
                                                                                                       '/'.join((self.expe.model, self.expe.corpus)),
                                                                                                       self._entropy, self.entropy_diff))

                    if self.expe.get('_write'):
                        self.write_current_state(self)
                        if mnb_num % 2000 == 0:
                            self.save(silent=True)
                            sys.stdout.flush()

