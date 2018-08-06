import sys, os
import inspect

# Model Manager Utilities
import numpy as np
from numpy import ma

# Frontend Manager Utilities
from .frontend import DataBase
from pymake import Model, Corpus,  GramExp
import pymake.io as io

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


    @classmethod
    def load(cls, expe):
        """ Return the frontend suited for the given expe"""

        corpus_name = expe.get('corpus') or expe.get('random') or expe.get('concept')
        if expe.get('driver'):
            corpus_name += '.' + expe.driver.strip('.')

        if '.' in corpus_name:
            c_split = corpus_name.split('.')
            c_name, c_ext = '.'.join(c_split[:-1]), c_split[-1]
        else:
            c_name = corpus_name
            c_ext = None

        _corpus = Corpus.get(c_name)
        if c_ext in GramExp._frontend_ext:
            # graph-tool object
            # @Todo: Corpus integration!
            if not _corpus:
                dt_lut = {'gt':'network'}
                _corpus = dict(data_type = dt_lut[c_ext])
            _corpus.update(data_format=c_ext)
        elif _corpus is False:
            raise ValueError('Unknown Corpus `%s\'!' % c_name)
        elif _corpus is None:
            return None

        if _corpus['data_type'] == 'text':
            from .frontendtext import frontendText
            frontend = frontendText(expe)
        elif _corpus['data_type'] == 'network':
            if _corpus.get('data_format') == 'gt':
                from .frontendnetwork import frontendNetwork_gt
                frontend = frontendNetwork_gt.from_expe(expe, corpus=_corpus)
            else:
                from .frontendnetwork import frontendNetwork
                # Obsolete loading design. @Todo
                frontend = frontendNetwork(expe)
                frontend.load_data(randomize=False)

        if hasattr(frontend, 'configure'):
            frontend.configure()

        return frontend


class ModelManager(object):
    """ Utility Class for Managing I/O and debugging Models

        Notes
        -----
        This class is more a wrapper or a **Meta-Model**.
    """

    log = logging.getLogger('root')

    def __init__(self, expe=None):
        self.expe = expe

    def is_model(self, m, _type):
        if _type == 'pymake':
             # __init__ method should be of type (expe, frontend, ...)
            pmk = inspect.signature(m).parameters.keys()
            score = []
            for wd in ('frontend', 'expe'):
                score.append(wd in pmk)
            return all(score)
        else:
            raise ValueError('Model type unkonwn: %s' % _type)

    def _get_model(self, frontend=None):
        ''' Get model with lookup in the following order :
            * pymake.model
            * mla
            * scikit-learn
        '''

        _model = Model.get(self.expe.model)
        if not _model:
            self.log.error('Model Unknown : %s' % (self.expe.model))
            raise NotImplementedError(self.expe.model)

        # @Improve: * initialize all model with expe
        #           * fit with frontend, transform with frontend (as sklearn do)
        if self.is_model(_model, 'pymake'):
            model = _model(self.expe, frontend)
        else:
            model = _model(self.expe)

        return model


    @classmethod
    def _load_model(cls, fn):

        _fn = io.resolve_filename(fn)
        if not os.path.isfile(_fn):
            # io integration?
            _fn += '.gz'

        if not os.path.isfile(_fn) or os.stat(_fn).st_size == 0:
            cls.log.error('No file for this model : %s' %_fn)
            cls.log.debug('The following are available :')
            for f in GramExp.model_walker(os.path.dirname(_fn), fmt='list'):
                cls.log.debug(f)
            return

        cls.log.info('Loading Model: %s' % fn)
        model = io.load(fn, silent=True)

        return model


    @staticmethod
    def update_expe(expe, model):
        ''' Configure some pymake settings if present in model. '''

        pmk_settings = ['_csv_typo', '_fmt']

        for _set in pmk_settings:
            if getattr(model, _set, None) and not expe.get(_set):
                expe[_set] = getattr(model, _set)


    @classmethod
    def from_expe(cls, expe, frontend=None, load=False):
    # frontend params is deprecated and will be removed soon...

        if load is False:
            mm = cls(expe)
            model = mm._get_model(frontend)
        else:
            fn = expe._output_path
            model = cls._load_model(fn)

        cls.update_expe(expe, model)

        return model


