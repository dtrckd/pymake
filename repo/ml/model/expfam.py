import numpy as np
import scipy as sp

from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma

from functools import partial, lru_cache

norm_pdf = sp.stats.norm.pdf

class ExpFamConjAbstract(object):

    ''' Base class fot conjugate exponential familly

        Attributes
        ----------
        _natex : list
            list of vectorized function to comptute the expeced natural parameters
            from the updated priors. (expected natural parameters equals partial derivative
            of the log partition.)
        _unin_priors : list
            list of uninformative priors (hyperparameters)

    '''

    def __init__(self):
        natfun = [getattr(self, a) for a in sorted(dir(self)) if (a.startswith('natex') and callable(getattr(self,a)))]
        self._natex = map(partial(np.vectorize, excluded=[1, 'ss']), natfun)

    def likelihood(self, *params, cache=250):
        ''' Returns a function tha compute data likelihood with caching options'''
        pass

    def ss(self, x):
        ''' Compute sufficient statistics vector '''
        pass

    def random_init(self, shape):
        ''' Returns an randomly initialied matrix with given shape
            using the curent distribution.
        '''
        pass

    def predict_edge(self, pp):
        pass


class Bernoulli(ExpFamConjAbstract):
    pass

class Normal(ExpFamConjAbstract):
    def __init__(self):
        super().__init__()

        self._unin_priors = np.array([1,3,1])

    @lru_cache(maxsize=200, typed=False)
    def ss(self, x):
        return np.asarray([x, x**2, 1])

    def expected_posterior(self, nat):
        m = nat[0] / nat[2]
        v = ((nat[2]+1)/2 * 2* nat[2] / (nat[1]*nat[2] - nat[0]**2))
        self.params = [m,v]
        return self.params

    def likelihood(self, cache=200):

        @lru_cache(maxsize=cache, typed=False)
        def compute(x):
            return norm_pdf(x, *self.params)
        #_likelihood =  defaultdict2(lambda x : sp.stats.norm.pdf(x, m, v)) # caching !
        return compute

    def random_init(self, shape):
        mat = np.random.normal(1, 1, size=np.prod(shape)).reshape(shape)
        # variance from a gamma ?
        return mat

    def predict_edge(self, theta, phi, pp,  data):
        cdf = sp.stats.norm.cdf(1, *self.params)
        probas = [(1 - theta[i].dot(cdf).dot(theta[j])) for i,j,_ in data]
        return np.asarray(probas)



    def natex1(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = t1*(t3+1)/(t2*t3 - t1**2)
        return tau

    def natex2(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = t3*(t3+1)/(2*(t2*t3 - t1**2))
        return tau

    def natex3(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = -0.5*((t1**2+t2)/(t2*t3-t1**2) - psi((t3+1)/0.5) -np.log(2*t3/(t2*t3-t1**2)))
        return tau


        #if kernel == 'bernoulli':
        #    self._unin_priors = np.array([0.5, 0.5])
        #    self._phi = np.random.beta(*self._unin_priors, size=K**2).reshape(K,K)

        #    x = frontend.weights()
        #    T = np.asarray([x, 1])
        #elif kernel == 'normal':

        #    def natex1(pos, ss):
        #        k, l = np.unravel_index(pos, ss[0].shape)
        #        t1, t2, t3 = ss[:, k, l]
        #        tau = t1*(t3+1)/(t2*t3 - t1**2)
        #        return tau

        #    def natex2(pos, ss):
        #        k, l = np.unravel_index(pos, ss[0].shape)
        #        t1, t2, t3 = ss[:, k, l]
        #        tau = t3*(t3+1)/(2*(t2*t3 - t1**2))
        #        return tau

        #    def natex3(pos, ss):
        #        k, l = np.unravel_index(pos, ss[0].shape)
        #        t1, t2, t3 = ss[:, k, l]
        #        tau = -0.5*((t1**2+t2)/(t2*t3-t1**2) - psi((t3+1)/0.5) -np.log(2*t3/(t2*t3-t1**2)))
        #        return tau

        #    def ss(x):
        #        return np.asarray([x, x**2, 1])

        #    # Kernel expected natural parameters / log partition gradient
        #    self._natex = map(partial(np.vectorize, excluded=[1, 'ss']), [natex1, natex2, natex3])
        #    # kernel sufficient statistics
        #    self._ss = ss
        #    # Kernel Likelihood
        #    self._ll = defaultdict2(lambda x,m,v : sp.stats.norm.pdf(x, m, v)) # caching !
        #    # Hyperpriors
        #    self._unin_priors = np.array([1,3,1])

        #    # Random Initialization
        #    self._phi = np.random.normal(1, 1, size=K**2).reshape(K,K)
        #    # variance from a gamma ?

        #elif kernel == 'poisson':
        #    self._unin_priors = np.array([1, 1])
        #    self._phi = np.random.gamma(*self._unin_priors, size=K**2).reshape(K,K)

        #    x = frontend.weights()
        #    T = np.asarray([x, 1])
        #else:
        #    raise NotImplementedError('kernel unknown: %s' % kernel)

class Poisson(ExpFamConjAbstract):
    pass

