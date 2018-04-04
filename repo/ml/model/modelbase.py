import inspect
from importlib import import_module
import sys
import re
from time import time
import pickle
from copy import deepcopy
import logging

import numpy as np
import scipy as sp

import pymake.io as io


#import sppy


# Todo: rethink sampler and Likelihood class definition.

def mmm(fun):
    # Todo / wrap latent variable routine
    # text vs networks type of data ?
    # * likelihood ?
    # * simlaritie ?
    return fun

class ModelBase():
    """"  Root Class for all the Models.

    * Suited for unserpervised model
    * Virtual methods for the desired propertie of models
    """

    __abstractmethods__ = 'model'

    default_settings = {
        '_write' : False,
        '_csv_typo' : None,
        '_fmt' : None, # unused...
        'iterations' : 3,
        'snapshot_freq': 42,
        'burnin' :  5, # (inverse burnin, last sample to keep
        'thinning' : 1,
    }

    log = logging.getLogger('root')

    def __init__(self, expe=None, frontend=None):
        """ Model Initialization strategy:
            1. self lookup from child initalization
            2. kwargs lookup
            3. default value
        """

        self.expe = expe
        self.frontend = frontend

        # change to semantic -> update value (t+1)
        self.samples = [] # actual sample
        self._samples    = [] # slice to save to avoid writing disk a each iteratoin. (ref format.write_current_state.)

        for k, v in self.default_settings.items():
            self._set_default_settings(k, expe, v)

        # @debug Frontend integratoin !
        # dev a Frontend.get_properties
        if hasattr(self.frontend, 'is_symmetric'):
            self._is_symmetric = self.frontend.is_symmetric()
        if hasattr(self.frontend, 'data_ma'):
            self.mask = self.frontend.data_ma.mask
        #np.fill_diagonal(self.frontend.data_ma, np.ma.masked)

        self._purge_objects = ['frontend', 'data_A', 'data_B']
        if hasattr(self, '_purge'):
            self._purge_objects.extend(self._purge)


    def _set_default_settings(self, key, expe, default):
        if key in expe:
            value = expe[key]
        elif hasattr(self, key):
            value = getattr(self, key)
        else:
            value = default

        return setattr(self, key, value)

    def _init(self, *args, **kwargs):
        ''' Init for fit method.
            Should initialize parmas that depend on the frontend/data.
        '''

        if hasattr(self, '_check_measures'):
            self._check_measures()

        if hasattr(self, '_init_params'):
            self._init_params(*args, **kwargs)

        self.begin_it = time()


    # @Dense mmsb
    def data_iter(self, data=None, randomize=False):
        ''' Iterate over various type of data :
            * ma.array (ignore masked
            * ndarray (todo ?)
            * What format for temporal/chunk/big data... ?
        '''
        return self._data_iter_ma(data, randomize)

    # @Dense mmsb
    def _data_iter_ma(self, data, randomize):
        if data is None:
            data_ma = self.frontend.data_ma
        else:
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

    def getK(self):
        theta, _ = self.get_params()
        return theta.shape[1]

    def getN(self):
        theta, _ = self.get_params()
        return theta.shape[0]

    def _init_params(self, *args, **kwargs):
        pass

    def _check_measures(self):
        if self.expe.get('deactivate_measures'):
            for m in self.expe.get('_csv_typo', '').split():
                if not hasattr(self, m):
                    setattr(self, m, None)

    def compute_measures(self):
        ''' Compute measure as model attributes.
            begin_it: is the time of the begining of the iteration.
        '''

        if self.expe.get('deactivate_measures'):
            return

        if hasattr(self, 'begin_it'):
            self.time_it = time() - self.begin_it

        params = self._reduce_latent()
        ll = self.likelihood(*params)
        kws = {'likelihood':ll}

        for meas in self.expe._csv_typo.split():

            if meas.lstrip('_') == 'entropy':
                old_entropy = getattr(self, '_entropy', -np.inf)
                _meas = self.compute_entropy(*params, **kws)
                self.entropy_diff = _meas - old_entropy
            elif hasattr(self, 'compute_'+meas.lstrip('_')):
                _meas = getattr(self, 'compute_'+meas.lstrip('_'))(*params, **kws)
            else:
                # Assume already computed
                getattr(self, meas) # raise exception if not here.
                continue

            setattr(self, meas, _meas)

        return



    @mmm #frontend
    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi
        likelihood = theta.dot(phi).dot(theta.T)
        return likelihood

    @mmm
    def similarity_matrix(self, theta=None, phi=None, sim='cos'):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        features = theta
        if sim in  ('dot', 'latent'):
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)
            sim = np.dot(features, features.T)/norm/norm.T
        elif sim in  ('model', 'natural'):
            sim = features.dot(phi).dot(features.T)
        else:
            self.log.error('Similaririty metric unknow: %s' % sim)
            sim = None

        if hasattr(self, 'normalization_fun'):
            sim = self.normalization_fun(sim)
        return sim


    def get_params(self):
        if hasattr(self, '_theta') and hasattr(self, '_phi'):
            return np.asarray(self._theta), np.asarray(self._phi)
        else:
            return self._reduce_latent()

    def _reduce_latent(self):
        ''' Estimate global parameters of a model '''
        raise NotImplementedError


    def update_hyper(self):
        self.log.warning('No method to update hyperparams..')
        return

    def get_hyper(self):
        self.log.error('no method to get hyperparams')
        return

    def save(self, silent=False):
        to_remove = []
        for k, v in self.__dict__.items():
            if hasattr(v, 'func_name') and v.func_name == '<lambda>':
                to_remove.append(k)
            if str(v).find('<lambda>') >= 0:
                # python3 hook, nothing better ?
                to_remove.append(k)
            #elif type(k) is defaultdict:
            #    setattr(self.model, k, dict(v))

        if to_remove or self._has_purge():
            model = deepcopy(self)
            model.purge()

            for k in to_remove:
                try:
                    delattr(model, k)
                except Exception as e:
                    self.log.debug('Cant delete object during model purging: %s' % e)
        else:
            model = self
            if hasattr(self, 'write_current_state'):
                delattr(self, 'write_current_state')


        fn = self.expe['_output_path']
        if not silent:
            self.log.info('Snapshotting Model : %s' % fn)
        else:
            sys.stdout.write('+')

        io.save(fn, model, silent=True)


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, deepcopy(v, memo))
            except Exception as e:
                self.log.debug('can\'t copy %s : %s. Passing on: %s' % (k, v, e))
                continue
        return result

    #@dense mmsb
    def get_mask(self):
        return self.mask

    #@dense mmsb
    def mask_probas(self, data):
        mask = self.get_mask()
        y_test = data[mask]
        p_ji = self.likelihood(*self._reduce_latent())
        probas = p_ji[mask]
        return y_test, probas

    #@dense mmsb
    def compute_roc(self):
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        data = self.frontend.data

        y_true, probas = self.mask_probas(data)
        fpr, tpr, thresholds = roc_curve(y_true, probas)

        roc = auc(fpr, tpr)
        return roc


    #@dense mmsb
    @mmm
    def predictMask(self, data, mask=True):
        self.log.info('Reducing latent variables...')

        if mask is True:
            masked = self.get_mask()
        else:
            masked = mask

        ### @Debug Ignore the Diagonnal when predicting.
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
            self.log.info('Finding Communities...')
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


    def fit(self, *args, **kwargs):
        ''' A core method.

            Template
            --------

            # cant take argument, they will be passed to
            # _init_params that you can safely  overwrite.
            #
            self._init()

            for _it in range(self.expe.iterations):
                # core process

                if self.expe.get('_write'):
                    self.write_current_state(self)

                    # In addition to that, model is automatically
                    # saved at the end of a script if the model
                    # is configured ie (called to load_model())
                    #
                    if _it > 0 and _it % self.snapshot_freq == 0:
                        self.save(silent=True)
                        sys.stdout.flush()
        '''
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    # Search ?


    def generate(self):
        raise NotImplementedError
    def get_clusters(self):
        raise NotImplementedError

    def _has_purge(self):
        return any([getattr(self, o, None) for o in self._purge_objects])

    def purge(self):
        for obj in self._purge_objects:
            if hasattr(self, obj):
                delattr(self, obj)



class ModelSkl(ModelBase):

    '''
    Notes
    -----
    Model class need to be serialisable. Module object are not serialisable.
    Avoid keeping it in the self object.
    '''

    def __init__(self, expe, frontend=None):
        super(ModelSkl, self).__init__(expe, frontend)

        # Load Sklearn Model
        if not hasattr(self, 'module'):
            self.log.error('ModelSkl base class need a {module} name attribute. Exiting.')
            exit(42)

        _module, _model = self._mm_from_str(self.module)
        self._spec = self._spec_from_expe(_model)

        # Init Sklearn model
        self.model = _model(**self._spec)


    @staticmethod
    def _mm_from_str(module):
        _module = module.split('.')
        _module, model_name = '.'.join(_module[:-1]), _module[-1]
        module = import_module(_module)
        _model = getattr(module, model_name)
        return module, _model

    def _spec_from_expe(self, _model):
        ''' Set Sklearn parameters. '''

        model_params = list(inspect.signature(_model).parameters)
        spec = dict()
        spec_map = getattr(self, 'spec_map', {})
        default_spec = getattr(self, '_default_spec', {})
        for k in model_params:
            if k in list(self.expe)+list(spec_map):
                _k = spec_map.get(k, k)
                if _k in self.expe:
                    spec[k] = self.expe[_k]
            elif k in default_spec:
                spec[k] = default_spec[k]

        return spec

    def __getattr__(self, attr):
        ''' Propagate sklearn attribute.

            Notes
            -----
            __getatrr__ is call only if the attr doesn't exist...
        '''

        if not 'model' in self.__dict__:
            raise AttributeError

        attr = attr.partition('__hack_me_')[-1]
        return getattr(self.model, attr)

    def fit(self, *args, **kwargs):
        fun =  self.__hack_me_fit
        self.log.info("Fitting `%s' model with spec: %s" % (type(self), str(self._spec)))
        return fun(*args, **kwargs)

    def transform(self, *args, **kwargs):
        fun =  self.__hack_me_transform
        data = fun(*args, **kwargs)

        if hasattr(self, 'post_transform'):
            for module in self.post_transform:
                _m, _model = self._mm_from_str(module)
                spec = self._spec_from_expe(_model)
                model = _model(**spec)
                data = model.fit_transform(data)

        return data

    def fit_transform(self, *args, **kwargs):
        fun =  self.__hack_me_fit_transform
        return fun(*args, **kwargs)

    # @Obsolete ?
    def predict(self, *args, **kwargs):
        fun =  self.__hack_me_predict
        return fun(*args, **kwargs)

    # get_params()
    # set_params()


class GibbsSampler(ModelBase):
    ''' Implemented method, except fit (other?) concerns MMM type models :
        * LDA like
        * MMSB like

        but for other (e.g IBP based), method has to be has to be overloaded...
        -> use a decorator @mmm to get latent variable ...
    '''
    __abstractmethods__ = 'model'
    def __init__(self, expe, frontend):
        super(GibbsSampler, self).__init__(expe, frontend)

    @mmm
    def compute_measures(self, begin_it=0):

        if self.expe.get('deactivate_measures'):
            return

        self._entropy = self.compute_entropy()

        if hasattr(self, 'compute_entropy_t'):
            self._entropy_t = self.compute_entropy_t()
        else:
            self._entropy_t = np.nan

        if '_roc' in self.expe._csv_typo.split():
            self._roc = self.compute_roc()

        self._alpha = np.nan
        self._gmma= np.nan
        self.alpha_mean = np.nan
        self.delta_mean = np.nan
        self.alpha_var = np.nan
        self.delta_var= np.nan

        self.time_it = time() - begin_it

    def fit(self, *args, **kwargs):

        self._init()

        self._entropy = self.compute_entropy()
        print( '__init__ Entropy: %f' % self._entropy)

        for _it in range(self.iterations):
            self._iteration = _it

            ### Sampling
            begin_it = time()
            self.sample()

            if _it >= self.iterations - self.burnin:
                if _it % self.thinning == 0:
                    self.samples.append([self._theta, self._phi])

            self.compute_measures(begin_it)
            print('.', end='')
            self.log.info('iteration %d, %s, Entropy: %f \t\t K=%d  alpha: %f gamma: %f' % (_it, '/'.join((self.expe.model, self.expe.corpus)),
                                                                                    self._entropy, self._K,self._alpha, self._gmma))
            if self.expe.get('_write'):
                self.write_current_state(self)
                if _it > 0 and _it % self.snapshot_freq == 0:
                    self.save(silent=True)
                    sys.stdout.flush()

        ### Clean Things
        if not self.samples:
            self.samples.append([self._theta, self._phi])
        self._reduce_latent()
        self.samples = None # free space
        return


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


    # keep only the most representative dimension (number of topics) in the samples
    @mmm
    def _reduce_latent(self):
        theta, phi = list(map(list, zip(*self.samples)))
        ks = [ mat.shape[1] for mat in theta]
        bn = np.bincount(ks)
        k_win = np.argmax(bn)
        self.log.debug('K selected: %d' % k_win)

        ind_rm = []
        [ind_rm.append(i) for i, v in enumerate(theta) if v.shape[1] != k_win]
        for i in sorted(ind_rm, reverse=True):
            theta.pop(i)
            phi.pop(i)

        self.log.debug('Samples Selected: %d over %s' % (len(theta), len(theta)+len(ind_rm) ))

        self._theta = np.mean(theta, 0)
        self._phi = np.mean(phi, 0)
        self.K = self._theta.shape[1]
        return self._theta, self._phi



class SVB(ModelBase):

    '''online EM/SVB

        Note: this obsolete base class is suited for dense masked numpy array.
              Base class for dense matrix svb...
    '''

    __abstractmethods__ = 'model'

    def __init__(self, expe, frontend):
        super(SVB, self).__init__(expe, frontend)

        self.entropy_diff = 0

    def _update_chunk_nnz(self, groups):
        _nnz = []
        for g in groups:
            if self._is_symmetric:
                count = 2* len(g)
                #count =  len(g)
            else:
                count = len(g)
            _nnz.append(count)

        self._nnz_vector = _nnz

    def _get_chunk(self):
        chunk = self.expe.get('chunk', 100)

        if isinstance(chunk, (int, float)):
            return chunk
        elif isinstance(chunk, str):
            pass
        else:
            raise TypeError('Unknown chunk type: %s' % type(chunk))

        mean_nodes = np.mean(self._len['dims'])
        try:
            chunk_mode, ratio = chunk.split('_')
        except ValueError as e:
            return float(chunk)

        if chunk_mode == 'adaptative':
            if '.' in ratio:
                ratio = float(ratio)
            else:
                ratio = int(ratio)

            chunk = ratio * mean_nodes
        else:
            raise ValueError('Unknown chunk mode: %s' % chunk_mode)

        return chunk


    def fit(self, *args, **kwargs):
        ''' chunk is the number of row to threat in a minibach '''

        self._init()

        data_ma = self.frontend.data_ma
        groups = self.data_iter(data_ma, randomize=True)
        chunk_groups = np.array_split(groups, self.chunk_len)
        self._update_chunk_nnz(chunk_groups)

        self._entropy = self.compute_entropy()
        print( '__init__ Entropy: %f' % self._entropy)
        for _id_mnb, minibatch in enumerate(chunk_groups):

            self.mnb_size = self._nnz_vector[_id_mnb]

            begin_it = time()
            self.sample(minibatch)

            self.compute_measures(begin_it)
            print('.', end='')
            self.log.info('Minibatch %d/%d, %s, Entropy: %f,  diff: %f' % (_id_mnb+1, self.chunk_len, '/'.join((self.expe.model, self.expe.corpus)),
                                                                        self._entropy, self.entropy_diff))
            if self.expe.get('_write'):
                self.write_current_state(self)
                if _id_mnb > 0 and _id_mnb % self.snapshot_freq == 0:
                    self.save(silent=True)
                    sys.stdout.flush()

    def sample(self, minibatch):
        '''
        Notes
        -----
        * self.iterations is actually the burnin phase of SVB.
        '''

        # not good
        #self._timestep_a = 0

        # self.iterations is actually the burnin phase of SVB.
        for _it in range(self.iterations+1):
            self.log.debug('init timestep A ?')
            #self._timestep_a = 0 # test, not good
            self._iteration = _it
            burnin = (_it < self.iterations)
            np.random.shuffle(minibatch)
            for _id_token, iter in enumerate(minibatch):
                self._id_token = _id_token
                self._xij = self.frontend.data_ma[tuple(iter)]
                self.maximization(iter)
                self.expectation(iter, burnin=burnin)


            if self._iteration != self.iterations-1 and self.expe.get('_verbose', 20) < 20:
                self.compute_measures()
                self.log.debug('it %d,  ELBO: %f, elbo diff: %f \t K=%d' % (_it, self._entropy, self.entropy_diff, self._K))
                if self.expe.get('_write'):
                    self.write_current_state(self)


        # Update global parameters after each "minibatech of links"
        # @debug if not, is the time_step evolve differently for N_Phi and N_Theta.
        minibashed_finished = True
        if minibashed_finished:
            self._purge_minibatch()

        #if self.elbo_diff < self.limit_elbo_diff:
        #    print('ELBO get stuck during data iteration : Sampling useless, return ?!')



