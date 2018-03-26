
import abc
import numpy as np
import scipy as sp
import scipy.stats as stats
import math
import logging
lgg = logging.getLogger('root')

""" This code was modified from the code
originally written by Zhai Ke (kzhai@umd.edu)."""

class IBP(object):
    __metaclass__ = abc.ABCMeta

    """
    @param alpha_hyper_parameter: hyper-parameter for alpha sampling, a tuple defining the parameter for an inverse gamma distribution
    @param metropolis_hastings_k_new: a boolean variable, set to true if we use metropolis hasting to estimate K_new, otherwise use truncated gibbs sampling """
    def __init__(self, #real_valued_latent_feature=True,
                 alpha_hyper_parameter='',
                 metropolis_hastings_k_new=True):
        # initialize the hyper-parameter for sampling _alpha
        # a value of None is a gentle way to say "do not sampling _alpha"
        if alpha_hyper_parameter.startswith('auto'):
            self._alpha_hyper_parameter = (1., 1.)
        else:
            self._alpha_hyper_parameter = False

        self._metropolis_hastings_k_new = metropolis_hastings_k_new

        self._x_title = "X-matrix-"
        self._z_title = "Z-matrix-"
        self._hyper_parameter_vector_title = "Hyper-parameter-vector-"
        # to remove
        #super(IBP, self).__init__()

    """
    @param data: a NxD np data matrix
    @param alpha: IBP hyper parameter
    @param initializ_Z: seeded Z matrix """
    #@abc.abstractmethod
    def _initialize(self, frontend, alpha=1.0, initial_Z=None, KK=None):
        self._alpha = alpha

        # Data matrix
        #self._Y = self.center_data(data)
        self._Y = frontend.data_ma

        # Binary case
        Yd = self._Y.data
        Yd[Yd <= 0 ] = -1
        Yd[Yd > 0 ] = 1
        (self._N, self._D) = self._Y.shape

        if(initial_Z == None):
            # initialize Z from IBP(alpha)
            if KK is None:
                self._Z = self.initialize_Z()
            else:
                self._Z = self.initialize_Z_debug(KK=KK)
        else:
            self._Z = initial_Z

        assert(self._Z.shape[0] == self._N)

        # make sure Z matrix is a binary matrix
        assert(self._Z.dtype == np.int)
        assert(self._Z.max() == 1 and self._Z.min() == 0)

        # record down the number of features
        self._K = self._Z.shape[1]

        return


    """
    initialize latent feature appearance matrix Z according to IBP(alpha) """
    def initialize_Z(self, N=None, alpha=None, left_ordered=True):
        if N is None:
            N = self._N
        if alpha is None:
            alpha = self._alpha
        # initialize matrix Z recursively in IBP manner
        # Debug case
        Z = []
        while len(Z) == 0 or Z.shape[1] == 0:
            Z = np.ones((0, 0), dtype=int)
            for i in range(1, N + 1):
                # sample existing features
                # Z.sum(axis=0)/i: compute the popularity of every dish, computes the probability of sampling that dish
                sample_dish = (np.random.uniform(0, 1, (1, Z.shape[1])) < (Z.sum(axis=0).astype(np.float) / i))
                # sample a value from the poisson distribution, defines the number of new features
                K_new = stats.poisson.rvs((alpha * 1.0 / i))
                # horizontally stack or append the new dishes to current object's observation vector, i.e., the vector Z_{n*}
                sample_dish = np.hstack((sample_dish, np.ones((1, K_new))))
                # append the matrix horizontally and then vertically to the Z matrix
                Z = np.hstack((Z, np.zeros((Z.shape[0], K_new))))
                Z = np.vstack((Z, sample_dish))

        if left_ordered:
            Z = self.leftordered(Z)

        assert(Z.shape[0] == N)
        return Z

    def initialize_Z_debug(self, N=None, alpha=None, KK=9):
        if N is None:
            N = self._N
        if alpha is None:
            alpha = self._alpha
        # initialize matrix Z recursively in IBP manner
        # Debug case
        Z = np.random.randint(0,2, (N, KK))
        #Z = np.ones((1, KK))
        #for i in xrange(2, N + 1):
        #    # sample existing features
        #    # Z.sum(axis=0)/i: compute the popularity of every dish, computes the probability of sampling that dish
        #    sample_dish = (np.random.uniform(0, 1, (1, Z.shape[1])) < (Z.sum(axis=0).astype(np.float) / i))
        #    # sample a value from the poisson distribution, defines the number of new features
        #    K_new = 0
        #    # horizontally stack or append the new dishes to current object's observation vector, i.e., the vector Z_{n*}
        #    sample_dish = np.hstack((sample_dish, np.ones((1, K_new))))
        #    # append the matrix horizontally and then vertically to the Z matrix
        #    Z = np.hstack((Z, np.zeros((Z.shape[0], K_new))))
        #    Z = np.vstack((Z, sample_dish))

        assert(Z.shape[0] == N)
        return Z

    """
    compute the log-likelihood of the Z matrix """
    def log_likelihood_Z(self):
        # compute {K_+} \log{\alpha} - \alpha * H_N, where H_N = \sum_{j=1}^N 1/j
        H_N = np.array([range(self._N)]) + 1.0
        H_N = np.sum(1.0 / H_N)
        log_likelihood = self._K * np.log(self._alpha) - self._alpha * H_N

        # compute the \sum_{h=1}^{2^N-1} \log{K_h!}
        Z_h = np.sum(self._Z, axis=0).astype(np.int)
        Z_h = list(Z_h)
        for k_h in set(Z_h):
            log_likelihood -= math.log(math.factorial(Z_h.count(k_h)))

        # compute the \sum_{k=1}^{K_+} \frac{(N-m_k)! (m_k-1)!}{N!}
        for k in range(self._K):
            m_k = Z_h[k]
            temp_var = 1.0
            if m_k - 1 < self._N - m_k:
                for k_prime in range(self._N - m_k + 1, self._N + 1):
                    if m_k != 1:
                        m_k -= 1

                    temp_var /= k_prime
                    temp_var *= m_k
            else:
                n_m_k = self._N - m_k
                for k_prime in range(m_k, self._N + 1):
                    temp_var /= k_prime
                    temp_var += n_m_k
                    if n_m_k != 1:
                        n_m_k -= 1

            log_likelihood += np.log(temp_var)

        return log_likelihood

    """
    sample alpha from conjugate posterior """
    def sample_alpha(self):
        assert(self._alpha_hyper_parameter != None)
        assert(type(self._alpha_hyper_parameter) == tuple)

        (alpha_hyper_a, alpha_hyper_b) = self._alpha_hyper_parameter

        K_plus = (self._Z.sum(axis=0) > 1).sum()
        posterior_shape = alpha_hyper_a + K_plus
        H_N = np.array([range(self._N)]) + 1.0
        H_N = np.sum(1.0 / H_N)
        posterior_rate = alpha_hyper_b + H_N # best convergence but longer (K get higher)
        #posterior_scale = alpha_hyper_b + self._N # LL lower but faster (K smaller)

        # Posterior in PyIBP ?
        #m = (self._Z != 0).astype(np.int).sum(axis=0)
        #posterior_shape = alpha_hyper_a + m.sum()
        #posterior_scale = float(1) / (alpha_hyper_b + self._N)

        alpha_new = np.random.gamma(posterior_shape, 1/posterior_rate, size=5).mean()
        lgg.debug('hyper sample: alpha: %s' % alpha_new )

        return alpha_new

    """
    sample standard deviation of a multivariant Gaussian distribution
    @param sigma_hyper_parameter: the hyper-parameter of the gamma distribution
    @param matrix: a r*c matrix drawn from a multivariant c-dimensional Gaussian distribution with zero mean and identity c*c covariance matrix """
    @staticmethod
    def sample_sigma(sigma_hyper_parameter, matrix):
        assert(sigma_hyper_parameter != None);
        assert(matrix != None);
        assert(type(sigma_hyper_parameter) == tuple);
        assert(type(matrix) == np.ndarray);

        (sigma_hyper_a, sigma_hyper_b) = sigma_hyper_parameter;
        (row, column) = matrix.shape;

        # compute the posterior_shape = sigma_hyper_a + n/2, where n = self._D * self._K
        posterior_shape = sigma_hyper_a + 0.5 * row * column;
        # compute the posterior_scale = sigma_hyper_b + sum_{k} (A_k - \mu_A)(A_k - \mu_A)^\top/2
        var = 0;
        if row >= column:
            var = np.trace(np.dot(matrix.transpose(), matrix));
        else:
            var = np.trace(np.dot(matrix, matrix.transpose()));

        posterior_scale = 1.0 / (sigma_hyper_b + var * 0.5);
        tau = stats.gamma.rvs(posterior_shape, scale=posterior_scale);
        sigma_a_new = np.sqrt(1.0 / tau);

        return sigma_a_new;

    def leftordered(self, Z=None):
        """ Returns the given matrix in left-ordered-form. """
        if Z is None:
            Z = self._Z
        l = list(Z.T)
        # tuple sorting: sort by first element, then second,etc
        l.sort(key=tuple)
        return np.array(l)[::-1].T

    """
    center the data, i.e., subtract the mean """
    @staticmethod
    def center_data(data):
        (N, D) = data.shape
        data = data - np.tile(data.mean(axis=0), (N, 1))
        return data

    def inititalize_stickbreaking_Z(customers, alpha=10, reducedprop=1.):
        """ Simple implementation of the Indian Buffet Process. Generates a binary matrix with
        customers rows and an expected number of columns of alpha * sum(1,1/2,...,1/customers).
        This implementation uses a stick-breaking construction.
        An additional parameter permits reducing the expected number of times a dish is tried. """
        # max number of dishes is distributed according to Poisson(alpha*sum(1/i))
        _lambda = alpha * np.sum(1. / np.array(list(range(1, customers + 1))))
        alpha /= reducedprop
        # we give it 2 standard deviations as cutoff
        maxdishes = int(_lambda + np.sqrt(_lambda) * 2) + 1
        res = np.zeros((customers, maxdishes), dtype=bool)
        stickprops = np.beta(alpha, 1, maxdishes) # nu_i
        currentstick = 1.
        dishesskipped = 0
        for i, nu in enumerate(stickprops):
            currentstick *= nu
            dishestaken = np.rand(customers) < currentstick * reducedprop
            if np.sum(dishestaken) > 0:
                res[:, i - dishesskipped] = dishestaken
            else:
                dishesskipped += 1
                return res[:, :maxdishes - dishesskipped]

    def plot_matrices(self):
        from plot import ColorMap
        import matplotlib.pyplot as plt
        # define parameter settings
        ps = [(0.1,), (1,), (10,) ]
        N = 30
        generateIBP = self.initialize_Z
        # generate a few matrices, on for each parameter setting
        ms = []
        for p in ps:
                m = generateIBP(N, p[0])
                ms.append(self.leftordered(m))
        # plot the matrices
        for m in ms:
            ColorMap(m, pixelspervalue=6)
        plt.draw()
