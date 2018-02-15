import sys, os
import inspect
import pickle, json # presence of this module here + in .frontend not zen

# Model Manager Utilities
import numpy as np
from numpy import ma

# Frontend Manager Utilities
from .frontend import DataBase
from .frontendtext import frontendText
from .frontendnetwork import frontendNetwork
from pymake import Model, Corpus, GramExp

import logging

class FrontendManager(object):
    """ Utility Class who aims at mananing/Getting the datastructure at the higher level.

        Parameters
        ----------
        get: return a frontend object.
        load: return a frontend object where data are
              loaded and filtered (sampled...) according to expe.
    """

    log = logging.getLogger('root')

    @staticmethod
    def get(expe, load=False):
        """ Return: The frontend suited for the given expe"""

        corpus_name = expe.get('corpus') or expe.get('random')

        _corpus = Corpus.get(corpus_name)
        if _corpus is None:
            raise ValueError('Unknown Corpus `%s\'!' % corpus)

        if _corpus['structure'] == 'text':
            frontend = frontendText(expe, load=load)
        elif _corpus['structure'] == 'network':
            frontend = frontendNetwork(expe, load=load)

        return frontend

    @classmethod
    def load(cls, expe):
        fr = cls.get(expe, load=True)
        fr.sample(expe.get('N'), randomize=False)
        return fr


# it is more a wrapper
class ModelManager(object):
    """ Utility Class for Managing I/O and debugging Models
    """

    log = logging.getLogger('root')

    def __init__(self, expe=None, frontend=None, data_t=None):
        self.expe = expe

        self.hyperparams = expe.get('hyperparams', dict())

        # Initialize model
        #self.model = self._get_model(frontend, data_t)

    def _format_dataset(self, data, data_t):
        if data is None:
            return None, None

        testset_ratio = self.expe.get('testset_ratio')

        if 'text' in str(type(data)).lower():
            #if issubclass(type(data), DataBase):
            self.log.warning('check WHY and WHEN overflow in stirling matrix !?')
            print('debug why error and i get walue superior to 6000 in the striling matrix ????')
            if testset_ratio is None:
                data = data.data
                data_t = None
            else:
                data, data_t = data.cross_set(ratio=testset_ratio)
        elif 'network' in str(type(data)).lower():
            data_t = None
            if testset_ratio is None:
                self.log.warning("testset-ratio option options unknow, data won't be masked array")
                data = data.data
            else:
                data = data.set_masked(testset_ratio)
        else:
            raise NotImplementedError('Data not understood')


        return data, data_t

    def is_model(self, m,  _type):
        if _type == 'pymake':
             # __init__ method should be of type (expe, frontend, ...)
            pmk = inspect.signature(m).parameters.keys()
            score = []
            for wd in ('frontend', 'expe'):
                score.append(wd in pmk)
            return all(score)
        else:
            raise NotImplementedError


    def _get_model(self, frontend=None, data_t=None):
        ''' Get model with lookup in the following order :
            * pymake.model
            * mla
            * scikit-learn
        '''

        # Not all model takes data (Automata ?)
        data, data_t = self._format_dataset(frontend, data_t)
        # @debug not used !
        #
        #if data is None:
        #    frontend = frontendNetwork(self.expe)
        #    frontend.data = np.array([[0],[0]])
        #    frontend.data_ma = ma.array([[0],[0]])

        _model = Model.get(self.expe.model)
        if not _model:
            self.log.error('Model Unknown : %s' % (self.expe.model))
            raise NotImplementedError()

        ### Learn to match signature ?!?
        if self.is_model(_model, 'pymake'):
            model = _model(self.expe, frontend)
        else:
            model = _model(**self.expe)

        return model

    def fit(self, frontend=None):
        ''' if frontend is not None, create a new model instance.
            This is a batch mode. Future will be a online update

            Parameters
            ----------
            frontend : dataBase
        '''

        if not hasattr(self, 'model'):
            self.model = self._get_model(frontend)

        if hasattr(self.model, 'fit'):
            fun = getattr(self.model, 'fit')
            fnargs = GramExp.sign_nargs(fun)
            if fnargs <= 0:
                # pymake
                fun()
            elif fnargs == 1:
                # sklearn type
                fun(frontend.data)
            else:
                raise NotImplementedError('Pipeline to model got unknown sinature')


        return

    # @obsolete / pmk compute
    # frontend ? no, data stat should be elsewhere.
    # Accept new data for prediction (now is just test data)
    def predict(self, frontend=None):
        if not hasattr(self.model, 'predict'):
            print('No predict method for self._name_ ?')
            return

        # @data_t manage mask vs held out
        # model don't necessarly own data...
        #if self.data_t == None and not hasattr(self.data, 'mask') :
        #    print('No testing data for prediction ?')
        #    return

        ### Prediction Measures
        data = frontend.data

        # if modelNetwork ...
        res = self.model.predictMask(data)
        #elif modelText

        ### Data Measure
        if frontend is not None:
            data_prop = frontend.get_data_prop()
            res.update(data_prop)

        if self.expe.write:
            frontend.save_json(res)
        else:
            self.log.debug(res)

    def initialization_test(self):
        ''' Measure perplexity on different initialization '''
        niter = 2
        pp = []
        likelihood = self.model.s.zsampler.likelihood
        for i in range(niter):
            self.model.s.zsampler.estimate_latent_variables()
            pp.append( self.model.s.zsampler.perplexity() )
            self.model = self.loadgibbs(expe.model, likelihood)

        np.savetxt('t.out', np.log(pp))

    @classmethod
    def _load_model(cls, fn):

        if not os.path.isfile(fn) or os.stat(fn).st_size == 0:
            cls.log.error('No file for this model : %s' %fn)
            cls.log.debug('The following are available :')
            for f in GramExp.model_walker(os.path.dirname(fn), fmt='list'):
                cls.log.debug(f)
            return None
        cls.log.info('Loading Model: %s' % fn)
        with open(fn, 'rb') as _f:
            try:
                model =  pickle.load(_f)
            except:
                # python 2to3 bug
                _f.seek(0)
                try:
                    model =  pickle.load(_f, encoding='latin1')
                except OSError as e:
                    cls.log.critical("Unknonw error while opening  while `_load_model' at file: %s" % (fn))
                    cls.log.error(e)
                    return
        return model


    @classmethod
    def from_file(cls, fn):
        return cls._load_model(fn)

    @classmethod
    def from_expe(cls, expe, init=False):
        if init is True:
            mm = cls(expe)
            model = mm._get_model()
        else:
            fn = GramExp.make_output_path(expe, 'pk')
            model = cls.from_file(fn)
        return model

    @classmethod
    def from_name(cls, expe):
        ''' Don't initialize the model (external facilities) '''
        modelname = expe.model
        _model =  Model.get(modelname)
        if _model is None:
            cls.log.error('model unknown: %s' % modelname)
            exit(3)
        return _model

    @classmethod
    def from_expe_frontend(cls, expe, frontend):
        # urgh,
        # structure and workflow for streaming ? temporal ?
        meta_model = cls(expe=expe, frontend=frontend)
        cls.model = meta_model._get_model(frontend)
        return cls.model


