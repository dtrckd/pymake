import numpy as np
import scipy as sp
import scipy.stats #sp.stats fails if not
from scipy.special import gammaln
from numpy.random import dirichlet, gamma, poisson, binomial, beta
from pymake.util.math import lognormalize, categorical

import logging
lgg = logging.getLogger('root')


# lambda fail to find import if _stirling if not
# visible in the global scope.
import sympy
from sympy.functions.combinatorial.numbers import stirling
try:
    from pymake.util.compute_stirling import load_stirling
    _stirling_mat = load_stirling()
    STIRLING_LOADED = True
except Exception as e:
    print('Stirling matrix cant be loaded:')
    print('error: ', e)
    STIRLING_LOADED = False

class MSampler(object):

    if STIRLING_LOADED:
        stirling_mat = lambda  _, x, y : _stirling_mat[x, y]
    else:
        lgg.error('stirling.npy file not found, using sympy instead (MMSB_CGS model will be 100 time slower !)')
        stirling_mat = lambda  _,x,y : np.asarray([float(sympy.log(stirling(x, i, kind=1)).evalf()) for i in y])

    def __init__(self, zsampler):

        self.zsampler = zsampler
        self.get_log_alpha_beta = zsampler.get_log_alpha_beta
        self.count_k_by_j = zsampler.doc_topic_counts

        # We don't know the preconfiguration of tables !
        self.m = np.ones(self.count_k_by_j.shape, dtype=int)
        self.m_dotk = self.m.sum(axis=0)

    def sample(self):
        self._update_m()

        indices = np.ndenumerate(self.count_k_by_j)

        lgg.debug('Sample m...')
        for ind in indices:
            j, k = ind[0]
            count = ind[1]

            if count > 0:
                # Sample number of tables in j serving dishe k
                params = self.prob_jk(j, k)
                sample = categorical(params) + 1
            else:
                sample = 0

            self.m[j, k] = sample

        self.m_dotk = self.m.sum(0)
        self.purge_empty_tables()

        return self.m

    def _update_m(self):
        # Remove tables associated with purged topics
        for k in sorted(self.zsampler.last_purged_topics, reverse=True):
            self.m = np.delete(self.m, k, axis=1)

        # Passed by reference, but why not...
        self.count_k_by_j = self.zsampler.doc_topic_counts
        K = self.count_k_by_j.shape[1]
        # Add empty table for new fancy topics
        new_k = K - self.m.shape[1]
        if new_k > 0:
            lgg.debug( 'msampler: %d new topics' % (new_k))
            J = self.m.shape[0]
            self.m = np.hstack((self.m, np.zeros((J, new_k), dtype=int)))

    # Removes empty table.
    def purge_empty_tables(self):
        # cant be.
        pass

    def prob_jk(self, j, k):
        # -1 because table of current sample topic jk, is not conditioned on
        njdotk = self.count_k_by_j[j, k]
        if njdotk == 1:
            return np.ones(1)

        possible_ms = np.arange(1, njdotk) # +1-1
        log_alpha_beta_k = self.get_log_alpha_beta(k)
        alpha_beta_k = np.exp(log_alpha_beta_k)

        normalizer = gammaln(alpha_beta_k) - gammaln(alpha_beta_k + njdotk)
        log_stir = self.stirling_mat(njdotk, possible_ms)

        params = normalizer + log_stir + possible_ms*log_alpha_beta_k

        return lognormalize(params)

class BetaSampler(object):

    def __init__(self, gmma, msampler):
        self.gmma = gmma
        self.msampler = msampler

        # Initialize restaurant with just one table.
        self.beta = dirichlet([1, gmma])

    def sample(self):
        lgg.debug( 'Sample Beta...')
        self._update_dirichlet_params()
        self.beta = dirichlet(self.dirichlet_params)

        return self.beta

    def _update_dirichlet_params(self):
        m_dotk_augmented = np.append(self.msampler.m_dotk, self.gmma)
        lgg.debug( 'Beta Dirichlet Prior: %s, alpha0: %.4f ' % (m_dotk_augmented, self.msampler.zsampler.alpha_0))
        self.dirichlet_params = m_dotk_augmented
