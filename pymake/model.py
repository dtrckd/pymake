import sys
from time import time
import inspect
from importlib import import_module
from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
import scipy as sp

import pymake.io as io
from pymake import logger

from sklearn.pipeline import make_pipeline


class ModelBase(object):
    """"  Root Class for all the Models.

    * Suited for unserpervised model
    * Virtual methods for the desired propertie of models
    """

    __abstractmethods__ = 'model'

    default_settings = {
        '_write' : False,
        '_measures' : [],
        '_fmt' : [], # unused...
        'iterations' : 3,
        'snapshot_freq': 42,
        'burnin' :  5, # (inverse burnin, last sample to keep
        'thinning' : 1,
    }

    log = logger

    def __init__(self, expe=None, frontend=None):
        """ Model Initialization strategy:
            1. self lookup from child initalization
            2. kwargs lookup
            3. default value
        """

        self.expe = expe
        self.frontend = frontend

        self._name = self.__class__.__name__.lower()

        # change to semantic -> update value (t+1)
        self.samples = [] # actual sample
        self._samples    = [] # slice to save to avoid writing disk a each iteratoin. (ref format.write_current_state.)

        for k, v in self.default_settings.items():
            self._set_default_settings(k, expe, v)

        self._typo_kws = self._extract_typo_kws()
        self._meas_kws = self._extract_meas_kws()
        self.measures = {}
        self._measure_cpt = 0

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
        ''' Iterate over various type of data:
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
            for m in self.expe.get('_measures', []):
                if not hasattr(self, m):
                    setattr(self, m, None)

    def _extract_typo_kws(self):

        if self.expe.get('_measures'):
            measures = self.expe._measures
        else:
            return {}

        kws = defaultdict(list)
        for param in measures:
            _param = ''.join(param.split('@')[1:])
            if not _param: continue
            for item in _param.split('&'):
                k, v = item.split('=')
                kws[k].append(v)

        return kws

    def _extract_meas_kws(self):

        meas_kws = OrderedDict()

        for _meas in self.expe._measures:
            kws = {}
            if '@' in _meas:
                meas, params = _meas.split('@')
                for param in params.split('&'):
                    k, v = param.split('=')
                    try:
                        kws[k] = int(v)
                    except ValueError as e:
                        kws[k] = v
            else:
                meas = _meas

            meas_kws[meas] = kws

        return meas_kws

    def compute_measures(self):
        ''' Compute measure as model attributes.
            begin_it: is the time of the begining of the iteration.
        '''

        if self.expe.get('deactivate_measures'):
            return

        if hasattr(self, 'begin_it'):
            self.time_it = time() - self.begin_it

        params = self._reduce_latent()

        # Measures preprocessing
        # @ml model dependancy here
        precomp = {}
        for data in self._typo_kws.get('data', []):

            if data == 'valid':
                ll = self.likelihood(*params, data=data)
                pp = None
            elif data == 'test':
                ll = None
                pp = self.posterior(*params, data=data)
            else:
                raise ValueError

            key = 'data_' + data
            precomp[key] = {'d': getattr(self, key),
                            'll': ll,
                            'pp': pp, }

        for meas, kws in self._meas_kws.items():

            if 'data' in kws:
                # @ml model dependancy here
                kws = kws.copy()
                v = kws['data']
                key = '_'.join(('data',v))
                if key in precomp:
                    # @ml model dependency
                    kws['data'] = precomp[key]

            if 'measure_freq' in kws:
                if self._measure_cpt % kws['measure_freq'] != 0:
                    continue

            if hasattr(self, 'compute_'+meas):
                _meas = getattr(self, 'compute_'+meas)(*params, **kws)
            else:
                # Assume already computed
                _meas = getattr(self, meas) # raise exception if not here.

            # set value and last diff
            self.measures[meas] = (_meas, _meas-self.measures.get(meas,[0])[0])

        self._measure_cpt += 1
        return self._measure_cpt



    #@mmm #frontend
    def likelihood(self, theta=None, phi=None):
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi
        likelihood = theta.dot(phi).dot(theta.T)
        return likelihood

    #@mmm
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
            self.log.error('Similaririty metric unknown: %s' % sim)
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
            self.log.info('Snapshotting Model: %s' % fn)
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
                self.log.debug('can\'t copy %s: %s. Passing on: %s' % (k, v, e))
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
    #@mmm
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

    ''' Wrapper around scikit-learn models.
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

        sk_modules = []

        if isinstance(self.module, str):
            sk_modules = [self.module]
        elif isinstance(self.module, list):
            sk_modules = self.module
        else:
            raise ValueError('Slearn model type unknown: %s' % (type(self.module), self.module))

        self._specs = []
        self._models = []

        model_names = self._name.split('-')
        assert(len(model_names) == len(sk_modules))

        for model_name, module in zip(model_names, sk_modules):
            _module, _model = self._mm_from_str(module)

            spec = self._spec_from_expe(_model, model_name)
            model = _model(**spec)

            self._specs.append(spec)
            self._models.append(model)

        # Init Sklearn model
        self.model = make_pipeline(*self._models)


    @staticmethod
    def _mm_from_str(module):
        _module = module.split('.')
        _module, model_name = '.'.join(_module[:-1]), _module[-1]
        module = import_module(_module)
        _model = getattr(module, model_name)
        return module, _model

    def _spec_from_expe(self, _model, model_name=None):
        ''' Set Sklearn parameters. '''

        if model_name is None:
            model_name = _model.__name__.split('.')[-1].lower()
        else:
            model_name = model_name.lower()
            # @debug model resolve name !
            model_name = model_name.split('.')[-1]

        model_params = list(inspect.signature(_model).parameters)
        spec = dict()
        spec_map = getattr(self, 'spec_map', {})
        default_spec = getattr(self, '_default_spec', {})

        model_spec = {}
        for k, v in self.expe.items():
            if k.find('__') >= 0:
                model, param = k.split('__')
                if model.lower() == model_name:
                    model_spec[param] = v

        for k in model_params:
            if k in list(model_spec)+list(spec_map):
                _k = spec_map.get(k, k)
                if _k in model_spec:
                    spec[k] = model_spec[_k]
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

        # or should it be hook_me ;)
        attr = attr.partition('__hack_me_')[-1]
        return getattr(self.model, attr)

    def fit(self, *args, **kwargs):
        fun =  self.__hack_me_fit
        self.log.info("Fitting `%s' model with spec: %s" % (type(self), str(self._specs)))
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

