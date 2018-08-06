from time import time
import sys
import numpy as np
import scipy as sp
from numpy import ma
import scipy.stats

from pymake.util.utils import defaultdict2
from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model.modelbase import SVB

#import warnings
#warnings.filterwarnings('error')
#warnings.catch_warnings()
##np.seterr(all='print')



class iwmmsb_scvb3(SVB):

    _purge = ['_kernel', '_lut_nbinom', '_likelihood']

    def _init_params(self, frontend):
        self.frontend = frontend

        # Save the testdata
        if hasattr(self.frontend, 'data_test'):
            data_test = frontend.data_test_w

            N = frontend.num_nodes()
            valid_ratio = frontend.get_validset_ratio() *2 # Because links + non_links...
            n_valid = np.random.choice(len(data_test), int(np.round(N*valid_ratio / (1+valid_ratio))), replace=False)
            n_test = np.arange(len(data_test))
            n_test[n_valid] = -1
            n_test = n_test[n_test>=0]
            self.data_test = data_test[n_test]
            self.data_valid = data_test[n_valid]

        # Data statistics
        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = frontend.num_nodes()
        _len['E'] = frontend.num_edges()
        _len['nnz'] = frontend.num_nnz()
        #_len['nnz_t'] = frontend.num_nnz_t()
        _len['dims'] = frontend.num_neighbors()
        _len['nnz_ones'] = frontend.num_edges()
        _len['nnzsum'] = frontend.num_nnzsum()
        self._len = _len

        self._K = self._len['K']
        self._is_symmetric = frontend.is_symmetric()

        # Eta init
        self._eta = []
        self._eta_limit = 1e-3
        self._eta_control = np.nan
        self._eta_count_init = 25
        self._eta_count = self._eta_count_init

        self._init_gradient()

        # Hyperparams
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        self.hyper_phi = np.asarray(self.expe['delta'])
        hyper_phi = self.expe['delta']


        self._random_ss_init()

        shift = self.expe.get('shift_w')
        #
        # Warn: Instable Part
        #
        if hyper_phi == 'auto':
            K = self._K
            N = self._len['N']
            self._hyper_phi = 'auto'

            #weights = np.squeeze(self.data_test[:,2].T)
            #mean = weights.mean()
            #var = weights.var()
            #density = frontend.density()
                                                          # roc5v | roc6v
            self.c0 = float(self.expe.get('c0', 10))      # 20      5
            self.r0 = float(self.expe.get('r0', 1))     # 0.5     0.5
            self.ce = float(self.expe.get('ce', 100))     # 200     100
            self.eps = float(self.expe.get('eps', 1e-6))  # 1e-6    1e-6
            #self.c0 = float(self.expe.get('c0', 10)) #weights mean
            #self.r0 = float(self.expe.get('r0', 0.2))
            #self.ce = float(self.expe.get('ce', 10))
            #self.eps = float(self.expe.get('eps', 1e-6))

            self.c0_r0 = self.c0*self.r0
            self.ce_eps = self.ce*self.eps
            self.ce_minus_eps = self.ce*(1-self.eps)

            rk = np.ones((K,K)) * self.r0
            pk = np.ones((K,K)) * self.eps
            self.hyper_phi = np.array([rk, pk, pk])

            self.log.info('Optimizing hyper enabled.')
            #self.hyper_phi = np.array([1,1])
        elif shift:
            a = 10
            b = shift
            self.hyper_phi = np.array([a,b,b])
        elif len(hyper_phi) == 2:
            self.hyper_phi = np.asarray(list(hyper_phi)+[hyper_phi[1]])
            self._hyper_phi = 'fix'
        else:
            raise ValueError('hyper parmeter hyper_phi dont understood: %s' % hyper_phi)

    def _init_gradient(self):
        N = self._len['N']
        self._timestep_a = np.zeros(N)
        self.gstep_theta = np.zeros(N)
        self._timestep_b = 0
        self._timestep_c = 0

        self._chi_a = self.expe.get('chi_a', 5)
        self._tau_a = self.expe.get('tau_a', 10)
        self._kappa_a = self.expe.get('kappa_a', 0.9)
        self._chi_b = self.expe.get('chi_b', 1)
        self._tau_b = self.expe.get('tau_b', 100)
        self._kappa_b = self.expe.get('kappa_b', 0.9)

        self._update_gstep_theta(np.arange(len(self._timestep_a)))
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

        if self.expe.get('homo') == 'assortative':
            N_phi_d = np.diag(np.random.dirichlet([0.5]*K)) *nnz*3/4
            N_phi_d1 = np.diag(np.random.dirichlet([0.5]*(K-1)), 1) *nnz*1/8
            if self._is_symmetric:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnz*1/8
                du = np.diag(np.ones(K-1), 1)==1
                dl = np.diag(np.ones(K-1), -1)==1
                N_phi_d1[dl] = N_phi_d1[du]
            else:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnz*1/8
            N_phi = N_phi_d + N_phi_d1
            self.N_phi = ma.masked_where(N_phi==0, N_phi)

            N_Y_d = np.diag(np.random.dirichlet([0.5]*K)) *nnzsum*3/4
            N_Y_d1 = np.diag(np.random.dirichlet([0.5]*(K-1)), 1) *nnzsum*1/8
            if self._is_symmetric:
                N_Y_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnzsum*1/8
                du = np.diag(np.ones(K-1), 1)==1
                dl = np.diag(np.ones(K-1), -1)==1
                N_Y_d1[dl] = N_Y_d1[du]
            else:
                N_Y_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnzsum*1/8
            N_Y = N_Y_d + N_Y_d1
            self.N_Y = ma.masked_where(N_Y==0, N_Y)

        else:
            self.N_phi = np.random.dirichlet([0.5]*K**2).reshape(K,K) * nnz
            self.N_Y = np.random.dirichlet([0.5]*K**2).reshape(K,K) * nnzsum
            #self.N_Y = np.random.poisson(0.1, (K,K)) * N

            if self._is_symmetric:
                self.N_theta_left = self.N_theta_right
                self.N_phi = np.triu(self.N_phi) + np.triu(self.N_phi, 1).T
                self.N_Y = np.triu(self.N_Y) + np.triu(self.N_Y, 1).T


    def _update_gstep_theta(self, idxs):
        ''' Gradient converge for kappa _in (0.5,1] '''
        chi = self._chi_a
        tau = self._tau_a
        kappa = self._kappa_a

        self.gstep_theta[idxs] = chi / ((tau + self._timestep_a[idxs])**kappa)

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

        self._k = self.N_Y + self.hyper_phi[0]
        self._p = self.hyper_phi[2] /( self.hyper_phi[2] * self.N_phi + 1)
        #self._phi = lambda x:sp.stats.nbinom.pmf(x, k, 1-p)
        self._phi = self._k*self._p / (1-self._p)

        #mean = k*p / (1-p)
        #var = k*p / (1-p)**2

        return self._theta, self._phi

    def _reduce_one(self, i, j, xij, update_local=True, update_kernel=True):

        if update_local:
            if self._is_symmetric:
                self.pik = self.pjk = self.N_theta_left[i] + self.hyper_theta
                self.pjk = self.pik
            else:
                self.pik = self.N_theta_left[i] + self.hyper_theta
                self.pjk = self.N_theta_right[j] + self.hyper_theta

        if update_kernel:
            k = self.N_Y + self.hyper_phi[0]
            p = self.hyper_phi[2] /( self.hyper_phi[2] * self.N_phi + 1)
            # @debug: Some invalie values here sometime !!
            self._kernel = defaultdict2(lambda x:sp.stats.nbinom.pmf(x, k, 1-p))

            if self._hyper_phi == 'auto':
                N = len(self.pik)
                rk = self.hyper_phi[0]
                pk = self.hyper_phi[1]

                # Way 1
                #a = np.ones(self.N_phi.shape)*(self.c0_r0 -1)
                #a[self.N_Y < 1.461] += 1

                # Way 2
                a = self.c0_r0 + self.N_Y

                _pk = 1-pk
                _pk[_pk < 1e-100] = 1e-100
                b = 1/(self.c0 - (self.N_phi)*np.log(_pk))
                #rk = np.random.gamma(a, b)
                rk = a*b

                c = self.ce_eps + self.N_Y
                d = self.ce_minus_eps + rk*self.N_phi

                pk = np.random.beta(c, d)
                e_pk = c/(c+d)
                pk[pk < 1e-100] = 1e-100

                self.hyper_phi = [rk, pk, e_pk]

                #self._residu = np.array([sp.stats.gamma.pdf(rk, self.c0*self.r0, scale=1/self.c0), sp.stats.beta.pdf(pk, self.ce*self.eps, self.ce*(1-self.eps)) ])

        kernel = self._kernel[xij]

        # debug: Underflow
        kernel[kernel<=1e-300] = 1e-100
        #kernel = ma.masked_invalid(kernel)

        outer_kk = np.log(np.outer(self.pik, self.pjk)) + np.log(kernel) #+ np.log(self._residu).sum()

        return lognormalize(outer_kk.ravel())

    def _optimize_hyper(self):
        n, t, t = self.hyper_phi
        K = self.N_theta_left.shape[1]

        phi_mean = np.triu(self._phi) if self._is_symmetric else self._phi

        # debug: Underflow
        phi_sum = phi_mean.sum()
        #phi_mean[phi_mean<=1e-300] = 1e-300
        log_phi_sum = np.log(phi_mean).sum()

        K_len = (K*(K-1)/2 +K) if self._is_symmetric else K**2

        t = n * K_len / phi_sum

        # http://bariskurt.com/calculating-the-inverse-of-digamma-function/
        #s = log_phi_sum / K_len + np.log(t)
        #x0 = lambda x: -1/(x + sp.special.digamma(1)) if x<-2.22 else np.exp(x)+0.5

        s = -log_phi_sum / K_len - np.log(K_len/phi_sum)
        x0 = lambda x: (3 - x + np.sqrt((x-3)**2 + 24*x))/(12*x)
        dig = lambda x: np.log(x) - sp.special.digamma(x) -s
        n = float(sp.optimize.minimize(dig, x0(s)).x)

        self.hyper_phi = np.array([n,t,t])

    def get_mean_phi(self):
        return  self._k*self._p / (1-self._p)

    def get_var_phi(self):
        return self._k*self._p / (1-self._p)**2

    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        #_likelihood = defaultdict2(lambda x : sp.stats.poisson.pmf(x, phi))
        _likelihood = defaultdict2(lambda x:sp.stats.nbinom.pmf(x, self._k, 1-self._p))
        qijs = np.array([ theta[i].dot(_likelihood[xij]).dot(theta[j]) for i,j,xij in self.data_valid])

        self._likelihood = _likelihood
        #qijs = ma.masked_invalid(qijs)
        return qijs

    def compute_entropy(self, theta=None, phi=None, **kws):
        return self.compute_entropy_t(theta, phi, **kws)

    def compute_entropy_t(self, theta=None, phi=None, **kws):
        if not hasattr(self, 'data_test'):
            return np.nan

        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        #weights = self.data_test[:,2].T
        #ll = sp.stats.poisson.pmf(weights, pij)
        ll = pij

        ll[ll<=1e-300] = 1e-100
        # Log-likelihood
        ll = np.log(ll).sum()
        # Perplexity is 2**H(X).
        #
        self._eta.append(ll)
        return ll

    def compute_elbo(self, theta=None, phi=None, **kws):
        # how to compute elbo for all possible links weights, mean?
        return None

    def compute_roc(self, theta=None, phi=None, treshold=None, **kws):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        weights = np.squeeze(self.data_test[:,2].T)

        #treshold = treshold or 'mean_data'

        if treshold == 'mean_data':
            mean = weights[weights>0].mean()
            std = weights[weights>0].std()
            trsh = int(mean)
            #trsh = int(mean+std)
        elif treshold == 'mean_model':
            mean_, var_ = self.get_mean_phi(), self.get_var_phi()
            mean = mean_.mean()
            std = var_.mean()**0.5
            trsh = int(mean)
            #trsh = int(mean+std)
        else:
            trsh = int(self.expe.get('shift_w',1)) if treshold is None else treshold

        trsh = trsh if trsh>=0 else 1

        self._probas = np.array([1 - sum([theta[i].dot(self._likelihood[v]).dot(theta[j]) for v in range(trsh)]) for i,j,_ in self.data_test])

        y_true = weights.astype(bool)*1
        self._y_true = y_true

        fpr, tpr, thresholds = roc_curve(y_true, self._probas)
        roc = auc(fpr, tpr)
        return roc

    def compute_pr(self, *args, **kwargs):
        from sklearn.metrics import average_precision_score
        return average_precision_score(self._y_true, self._probas)

    def mask_probas(self, *args):
        # Copy of compute_roc
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        theta, phi = self._reduce_latent()

        weights = self.data_test[:,2]

        treshold = 'mean_model'
        if treshold == 'mean_data':
            mean = weights[weights>0].mean()
            std = weights[weights>0].std()
        if treshold == 'mean_model':
            mean_, var_ = self.get_mean_phi(), self.get_var_phi()
            mean = mean_.mean()
            std = var_.mean()**0.5

        #trsh = int(mean_w)
        trsh = int(mean+std)

        qijs = self.likelihood(theta, phi)
        y_true = weights.astype(bool)*1
        probas = 1 - sp.stats.poisson.cdf(trsh, qijs)

        return y_true, probas


    def compute_wsim(self, theta=None, phi=None, **kws):
        if 'likelihood' in kws:
            pij = kws['likelihood']
        else:
            if theta is None:
                theta, phi = self._reduce_latent()
            pij = self.likelihood(theta, phi)

        weights = self.data_test[:,2].T

        ws = np.array([ theta[i].dot(phi).dot(theta[j]) for i,j,w in self.data_test if w > 0])

        # l1 norm
        wd = weights[weights>0]
        nnz = len(wd)
        mean_dist = np.abs(ws - wd).sum() / nnz
        return mean_dist


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
        ''' chunk is the number of row to process in a minibach '''

        self._init(frontend)

        # Init sampling variables
        observed_pt = 0
        mnb_num = 0
        vertex = None

        self.BURNIN = 150
        qij_samples = []
        node_idxs = []
        weights = []
        _qijs_sum = 0
        _qijs_w_sum = 0
        _norm = 0

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
                new_mnb = True

                update_kernel = True
                update_local = True
                burnin = 0
            else:
                i = source
                j = target
                weights.append(weight)
                if direction == 0:
                    node_idxs.append(j)
                else:
                    node_idxs.append(i)

                # Maximization
                qij_samples.append( self._reduce_one(i,j, weight, update_local, update_kernel).reshape(self._len['K'], self._len['K']) )

                observed_pt += 1
                burnin += 1
                update_kernel = False
                update_local = False

            if (new_mnb or burnin % self.BURNIN == 0) and qij_samples:
                qijs = np.asarray(qij_samples)
                ## Update global gradient / Expectation
                ##norm=1
                norm = qijs.shape[0]
                qijs_sum = qijs.sum(0)

                _qijs_sum += qijs_sum
                if set_pos != '0':
                    _qijs_w_sum += np.sum([weights[n]*qijs[n] for n in range(len(weights)) ],0)
                _norm += norm

                gstep_v = self.gstep_theta[vertex]
                gstep_nodes = self.gstep_theta[node_idxs][None].T

                if direction == 0:
                    self.N_theta_left[i] = (1-gstep_v)*self.N_theta_left[i] + gstep_v*scaler*qijs_sum.sum(0) /norm
                    self.N_theta_right[node_idxs] = (1-gstep_nodes)*self.N_theta_right[node_idxs] + gstep_nodes*scaler*qijs.sum(2)
                else:
                    self.N_theta_left[node_idxs] = (1-gstep_nodes)*self.N_theta_left[node_idxs] + gstep_nodes*scaler*qijs.sum(1)
                    self.N_theta_right[j] = (1-gstep_v)*self.N_theta_right[j] + gstep_v*scaler*qijs_sum.sum(1) /norm

                self._timestep_a[vertex] += norm
                self._timestep_a[node_idxs] += 1
                self._update_gstep_theta([vertex]+node_idxs)

                qij_samples.clear()
                node_idxs.clear()
                weights.clear()

                update_local = True


            if new_mnb:
                if vertex is None:
                    # Enter here only once !%!
                    mnb_total = frontend.num_mnb()
                    self.begin_it = time()

                    set_pos = _set_pos
                    vertex = _vertex
                    direction = _direction
                    scaler = _scaler
                    new_mnb = False
                    continue


                self.N_phi = (1 - self.gstep_phi)*self.N_phi + self.gstep_phi * scaler * _qijs_sum /_norm
                if set_pos != '0':
                    self.N_Y = (1 - self.gstep_y)*self.N_Y + self.gstep_y * scaler * _qijs_w_sum /_norm
                    self._timestep_c += _norm
                    self._update_gstep_y()

                    #if self._hyper_phi == 'auto' and mnb_num % 50 == 0:
                    #    self._optimize_hyper()

                self._timestep_b += _norm
                self._update_gstep_phi()


                # Allocate current state variable
                set_pos = _set_pos
                vertex = _vertex
                direction = _direction
                scaler = _scaler

                _qijs_sum = 0
                _qijs_w_sum = 0
                _norm = 0

                new_mnb = False
                mnb_num += 1

                if mnb_num % (self.expe['zeros_set_len']*5) == 0:
                    prop_edge = observed_pt / self._len['nnz']
                    self._observed_pt = observed_pt
                    self.compute_measures()

                    print('.', end='')
                    self.log.info('it %d | prop edge: %.2f | mnb %d/%d, %s, Entropy: %f,  diff: %f' % (_it, prop_edge,
                                                                                                       mnb_num, mnb_total,
                                                                                                       '/'.join((self.expe.model, self.expe.corpus)),
                                                                                                       self._entropy, self.entropy_diff))

                    if self._check_eta():
                        break

                    if self.expe.get('_write'):
                        self.write_current_state(self)
                        if mnb_num % 4000 == 0:
                            self.save(silent=True)
                            sys.stdout.flush()


    def _check_eta(self):

        if self._eta_control is not np.nan:
            self._eta_count -=1
            if self._eta_count == 0:
                if self._eta[-1] - self._eta_control < self._eta_limit:
                    self.log.warning('Exit iteration cause eta criteria met.')
                    return True
                else:
                    self._eta_count = self._eta_count_init
                    self._eta_control = np.nan
                    self._eta = [self._eta[-1]]
        elif len(self._eta) > 10:
            if self._eta[-1] - self._eta[0] < self._eta_limit:
                self._eta_control = self._eta[-1]
                print('-', end='')
            else:
                self._eta = [self._eta[-1]]




class iwmmsb_scvb3_auto(iwmmsb_scvb3):
    def __init__(self, expe, frontend):

        if expe['delta'] != 'auto':
            raise ValueError("Delta should be `auto' for wmmsb_scvb_np model...")

        super().__init__(expe, frontend)

