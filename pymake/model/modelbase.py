# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import pickle
from copy import deepcopy
import re

import numpy as np
import scipy as sp
import scipy.stats #sp.stats fails if not
from scipy.special import gammaln
from numpy.random import dirichlet, gamma, poisson, binomial, beta

from pymake.util.math import lognormalize, categorical

#import sppy

import logging
lgg = logging.getLogger('root')

# Todo: rethink sampler and Likelihood class definition.

class ModelBase(object):
    """"  Root Class for all the Models.

    * Suited for unserpervised model
    * Virtual methods for the desired propertie of models
    """
    __abstractmethods__ = 'model'
    default_settings = {
        'write' : False,
        'output_path' : 'tm-output',
        '_csv_typo' : None,
        'fmt' : None,
        'iterations' : 1,
        'snapshot_freq': 20,
    }
    log = logging.getLogger('root')
    def __init__(self, **kwargs):
        """ Model Initialization strategy:
            1. self lookup from child initalization
            2. kwargs lookup
            3. default value
        """

        # change to semantic -> update value (t+1)
        self.samples = [] # actual sample
        self._samples    = [] # slice to save to avoid writing disk a each iteratoin. (ref format.write_it_step.)

        for k, v in self.default_settings.items():
            self._init(k, kwargs, v)


        # Why this the fuck ? to remove
        #super(ModelBase, self).__init__()

    def _init(self, key, kwargs, default):
        if hasattr(self, key):
            value = getattr(self, key)
        elif key in kwargs:
            value = kwargs[key]
        else:
            value = default

        return setattr(self, key, value)


    def similarity_matrix(self, theta=None, phi=None, sim='cos'):
        if theta is None:
            theta = self.theta
        if phi is None:
            phi = self.phi

        features = theta
        if sim in  ('dot', 'latent'):
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)
            sim = np.dot(features, features.T)/norm/norm.T
        elif sim in  ('model', 'natural'):
            sim = features.dot(phi).dot(features.T)
        else:
            lgg.error('Similaririty metric unknow: %s' % sim)
            sim = None

        if hasattr(self, 'normalization_fun'):
            sim = self.normalization_fun(sim)
        return sim

    def get_params(self):
        if hasattr(self, 'theta') and hasattr(self, 'phi'):
            return self.theta, self.phi
        else:
            return self._reduce_latent()

    def getK(self):
        theta, _ = self.get_params()
        return theta.shape[1]

    def getN(self):
        theta, _ = self.get_params()
        return theta.shape[0]

    def purge(self):
        """ Remove variable that are non serializable. """
        return

    def update_hyper(self):
        lgg.error('no method to update hyperparams')
        return

    def get_hyper(self):
        lgg.error('no method to get hyperparams')
        return

    def save(self, silent=False):
        fn = self.output_path + '.pk'
        model = deepcopy(self)
        model.purge()
        to_remove = []
        for k, v in model.__dict__.items():
            if hasattr(v, 'func_name') and v.func_name == '<lambda>':
                to_remove.append(k)
            if str(v).find('<lambda>') >= 0:
                # python3 hook, nothing better ?
                to_remove.append(k)
            #elif type(k) is defaultdict:
            #    setattr(self.model, k, dict(v))

        for k in to_remove:
            try:
                delattr(model, k)
            except:
                pass

        if not silent:
            lgg.info('Snapshotting Model : %s' % fn)
        with open(fn, 'wb') as _f:
            return pickle.dump(model, _f, protocol=pickle.HIGHEST_PROTOCOL)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, deepcopy(v, memo))
            except:
                #print('can\'t copy %s : %s' % (k, v))
                continue
        return result

    def get_mask(self):
        return self.mask

    def mask_probas(self, data):
        mask = self.get_mask()
        y_test = data[mask]
        p_ji = self.likelihood(*self.get_params())
        probas = p_ji[mask]
        return y_test, probas


    def predictMask(self, data, mask=True):
        lgg.info('Reducing latent variables...')

        if mask is True:
            masked = self.get_mask()
        else:
            masked = mask

        ### @Debug Ignore the Diagonnal
        np.fill_diagonal(masked, False)

        ground_truth = data[masked]

        p_ji = self.likelihood(*self.get_params())
        prediction = p_ji[masked]
        prediction = sp.stats.bernoulli.rvs( prediction )
        #prediction[prediction >= 0.5 ] = 1
        #prediction[prediction < 0.5 ] = 0

        ### Computing Precision
        test_size = float(ground_truth.size)
        good_1 = ((prediction + ground_truth) == 2).sum()
        precision = good_1 / float(prediction.sum())
        rappel = good_1 / float(ground_truth.sum())
        g_precision = (prediction == ground_truth).sum() / test_size
        mask_density = ground_truth.sum() / test_size

        ### Finding Communities
        if hasattr(self, 'communities_analysis'):
            lgg.info('Finding Communities...')
            communities = self.communities_analysis(data)
            K = self.K
        else:
            communities = None
            K = self.expe.get('K')

        res = {'Precision': precision,
               'Recall': rappel,
               'g_precision': g_precision,
               'mask_density': mask_density,
               'clustering':communities,
               'K': K
              }
        return res

    def fit(self):
        raise NotImplementedError
    # Just for MCMC ?():
    def predict(self):
        raise NotImplementedError

    # get_params ~ transform ?sklearn

    def generate(self):
        raise NotImplementedError
    def get_clusters(self):
        raise NotImplementedError

def mmm(fun):
    # Todo / wrap latent variable routine
    return fun

class GibbsSampler(ModelBase):
    ''' Implmented method, except fit (other?) concerns MMM type models :
        * LDA like
        * MMSB like

        but for other (e.g IBP based), method has to be has to be overloaded...
        -> use a decorator @mmm to get latent variable ...
    '''
    __abstractmethods__ = 'model'
    def __init__(self, sampler,  **kwargs):
        self.s = sampler
        super(GibbsSampler, self).__init__(**kwargs)

    @mmm
    def compute_measures(self):
        self._K = self.s.zsampler._K

        self._entropy = self.evaluate_entropy()
        if self.data_t is not None:
            self._entropy_t = self.predictive_likelihood()
        else:
            self._entropy_t = np.nan

        alpha_0 = self.s.zsampler.alpha_0
        try:
            gmma = self.s.betasampler.gmma
            alpha = np.exp(self.s.zsampler.log_alpha_beta)
        except:
            gmma = np.nan
            alpha = np.exp(self.s.zsampler.logalpha)

        self._alpha = alpha_0
        self._gmma = gmma
        self.alpha_mean = alpha.mean()
        self.alpha_var = alpha.var()
        self.delta_mean = self.s.zsampler.likelihood.delta.mean()
        self.delta_var = self.s.zsampler.likelihood.delta.var()


    def fit(self):

        lgg.info( '__init__  Entropy: %f' % self.evaluate_entropy())
        for _it in range(self.iterations):
            self._iteration = _it

            ### Sampling
            begin_it = datetime.now()
            self.s.sample()
            self.time_it = (datetime.now() - begin_it).total_seconds() / 60

            if _it >= self.iterations - self.burnin:
                if _it % self.thinning == 0:
                    self.samples.append([self._theta, self._phi])

            self.compute_measures()
            lgg.info('Iteration %d,  Entropy: %f \t\t K=%d  alpha: %f gamma: %f' % (_it, self._entropy, self._K,self._alpha, self._gmma))
            if self.write:
                self.write_it_step(self)
                if (_it > 0 and _it % self.snapshot_freq == 0) or (_it == self.iterations-1):
                    self.save(silent=(not _it == self.iterations-1))

        ### Clean Things
        print()
        if not self.samples:
            self.samples.append([self._theta, self._phi])
        return

    @mmm #frontend
    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self.theta
        if phi is None:
            phi = self.phi
        likelihood = theta.dot(phi).dot(theta.T)
        return likelihood


    @mmm
    def update_hyper(self, hyper):
        if hyper is None:
            return
        elif isinstance(type(hyper), (tuple, list)):
            alpha = hyper[0]
            gmma = hyper[1]
            delta = hyper[2]
        else:
            delta = hyper.get('delta')
            alpha = hyper.get('alpha')
            gmma = hyper.get('gmma')

        if delta:
            self._delta = delta
        if alpha:
            self._alpha = alpha
        if gmma:
            self._gmma = gmma

    @mmm
    def get_hyper(self):
        if not hasattr(self, '_alpha'):
            try:
                self._delta = self.s.zsampler.likelihood.delta
                if type(self.s) is NP_CGS:
                    self._alpha = self.s.zsampler.alpha_0
                    self._gmma = self.s.betasampler.gmma
                else:
                    self._alpha = self.s.zsampler.alpha
                    self._gmma = None
            except:
                lgg.error('Need to propagate hyperparameters to BaseModel class')
                self._delta = None
                self._alpha = None
                self._gmma =  None
        return self._alpha, self._gmma, self._delta

    # Nasty hack to make serialisation possible
    @mmm
    def purge(self):
        self.s.zsampler.betasampler = None
        self.s.zsampler._nmap = None
        self.s.msampler = None
        self.s.betasampler = None
        self.s.zsampler.likelihood = None

    @mmm
    def evaluate_entropy(self, data=None):
        self._theta, self._phi = self.s.zsampler.estimate_latent_variables()
        return self.s.zsampler.entropy(data)

    # keep only the most representative dimension (number of topics) in the samples
    @mmm
    def _reduce_latent(self):
        theta, phi = list(map(list, zip(*self.samples)))
        ks = [ mat.shape[1] for mat in theta]
        bn = np.bincount(ks)
        k_win = np.argmax(bn)
        lgg.debug('K selected: %d' % k_win)

        ind_rm = []
        [ind_rm.append(i) for i, v in enumerate(theta) if v.shape[1] != k_win]
        for i in sorted(ind_rm, reverse=True):
            theta.pop(i)
            phi.pop(i)

        lgg.debug('Samples Selected: %d over %s' % (len(theta), len(theta)+len(ind_rm) ))

        self.theta = np.mean(theta, 0)
        self.phi = np.mean(phi, 0)
        self.K = self.theta.shape[1]
        return self.theta, self.phi

# lambda fail to find import if _stirling if not
# visible in the global scope.
import sympy
from sympy.functions.combinatorial.numbers import stirling
try:
    from pymake.util.compute_stirling import load_stirling
    _stirling_mat = load_stirling()
    STIRLING_LOADED = True
except Exception as e:
    print(e)
    STIRLING_LOADED = False

class MSampler(object):

    if STIRLING_LOADED:
        stirling_mat = lambda  _, x, y : _stirling_mat[x, y]
    else:
        lgg.error('stirling.npy file not found, using sympy instead (it will be 100 time slower !)')
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
            lgg.info( 'msampler: %d new topics' % (new_k))
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

class SVB(ModelBase):

    '''online EM/SVB'''

    __abstractmethods__ = 'model'

    def __init__(self, expe, frontend=None):
        self.fmt = '%d %.4f %.8f %.8f %d'
        super(SVB, self).__init__(**expe)
        self.elbo = None
        self.limit_elbo_diff = 1e-3
        self.fr = frontend
        self.mask = self.fr.data_ma.mask
        self.expe = expe

        if expe:
            self._init_params(frontend)

    def _init_params(self, frontend):
        raise NotImplementedError

    def data_iter(self, batch, randomize=True):
        raise NotImplementedError

    def _update_chunk_nnz(self, groups):
        _nnz = []
        frontend = self.fr
        dama = self.fr.data_ma
        for g in groups:
            if frontend.is_symmetric():
                count = 2* len(g)
                #count =  len(g)
            else:
                count = len(g)
            _nnz.append(count)

        self._nnz_vector = _nnz

    def fit(self):
        ''' chunk is the number of row to threat in a minibach '''

        data_ma = self.fr.data_ma
        _abc = self.data_iter(data_ma, randomize=False)
        chunk_groups = np.array_split(_abc, self.chunk_len)
        self._update_chunk_nnz(chunk_groups)

        print('__init__ Entropy %f' % self.entropy())
        for _id_mnb, minibatch in enumerate(chunk_groups):

            self.mnb_size = self._nnz_vector[_id_mnb]

            begin_it = datetime.now()
            self.sample(minibatch)
            time_it = (datetime.now() - begin_it).total_seconds() / 60

            self.compute_measures()
            lgg.info('mnibatch %d/%d,  ELBO: %f,  elbo diff: %f' % (_id_mnb+1, self.chunk_len, self.elbo, self.elbo_diff))
            if self.expe.get('write'):
                self.write_it_step(self)
                if (_id_mnb > 0 and _id_mnb % self.snapshot_freq == 0) or (_id_mnb == len(chunk_groups)-1):
                    self.save(silent=(not _id_mnb == len(chunk_groups)-1))

    def sample(self, minibatch):
        '''
        Notes
        -----
        * self.iterations is actually the burnin phase of SVB.
        '''

        # self.iterations is actually the burnin phase of SVB.
        for _it in range(self.iterations+1):
            self._iteration = _it
            burnin = (_it < self.iterations)
            #np.random.shuffle(minibatch)
            for _id_token, iter in enumerate(minibatch):
                self._id_token = _id_token
                self._xij = self.fr.data_ma[tuple(iter)]
                self.maximization(iter)
                self.expectation(iter, burnin=burnin)


                #self.entropy()
                #print(self._pp)

            self.compute_measures()
            if self._iteration != self.iterations-1 and self.expe.verbose < 20:
                lgg.debug('it %d,  ELBO: %s, elbo diff: %s' % (_it, self.elbo, self.elbo_diff))
                if self.expe.get('write'):
                    self.write_it_step(self)


        # Update global parameters after each "minibatech of links"
        # @debug if not, is the time_step evolve differently for N_Phi and N_Theta.
        minibashed_finished = True
        if minibashed_finished:
            self._purge_minibatch()
            self.inc_time()

        #if self.elbo_diff < self.limit_elbo_diff:
        #    print('ELBO get stuck during data iteration : Sampling useless, return ?!')


    def compute_measures(self):
        self.update_elbo()
        self._entropy_t = 0
        self._K = self.N_phi.shape[1]


    def update_elbo(self):
        ## Get real ELBO instead of ll
        nelbo = self.entropy()
        self.elbo_diff = nelbo - self.elbo
        self.elbo = nelbo
        return self.elbo

    def get_elbo(self):
        raise NotImplementedError

    def maximization(self):
        raise NotImplementedError

    def expectation(self):
        raise NotImplementedError

    def _purge_minibatch(self):
        raise NotImplementedError

    def purge(self):
        self.fr = None
