# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import logging
lgg = logging.getLogger('root')

# Frontend Manager Utilities
from .frontend import DataBase
from .frontendtext import frontendText
from .frontendnetwork import frontendNetwork
from .frontend_io import *

# Model Manager Utilities
import numpy as np
from numpy import ma
import pickle, json # presence of this module here + in .frontend not zen


# This is a mess, hormonize things with models
from pymake.model.hdp import mmsb, lda
from pymake.model.ibp.ilfm_gs import IBPGibbsSampling


# __future__ **$ù*$ù$
#### @Debug/temp modules name changed in pickle model
from pymake.model import hdp, ibp
sys.modules['hdp'] = hdp
sys.modules['ibp'] = ibp
import pymake.model
sys.modules['models'] = pymake.model


try:
    sys.path.insert(1, '../../gensim')
    import gensim as gsm
    from gensim.models import ldamodel, ldafullbaye
    Models = {'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}
except:
    pass

class FrontendManager(object):
    """ Utility Class who aims at mananing/Getting the datastructure at the higher level.
    """
    @staticmethod
    def get(config, load=False):
        """ Return: The frontend suited for the given configuration"""
        corpus = config.get('corpus_name') or config.get('corpus') or config.get('random')
        corpus_typo = {
            'network': [
                'clique', 'alternate', 'BA', # random
                'facebook',
                'generator',
                'bench',
                'fb_uc',
                'manufacturing',
                'propro',
                'blogs',
                'euroroad',
                'emaileu'
            ],
            'text': ['reuter50',
                     'nips12',
                     'nips',
                     'enron',
                     'kos',
                     'nytimes',
                     'pubmed',
                     '20ngroups',
                     'odp',
                     'wikipedia',
                     'lucene']}

        frontend = None
        for key, cps in corpus_typo.items():
            if corpus.startswith(tuple(cps)):
                if key == 'text':
                    frontend = frontendText(config, load=load)
                    break
                elif key == 'network':
                    frontend = frontendNetwork(config, load=load)
                    break

        if frontend is None:
            raise ValueError('Unknown Corpus `%s\'!' % corpus)
        return frontend


# it is more a wrapper
class ModelManager(object):
    """ Utility Class for Managing I/O and debugging Models
    """
    def __init__(self, config=None, data=None, data_t=None):
        self._init(config)

        if self.cfg.get('load_model'):
            return
        if not self.model_name:
            return

        # @frontend
        if data is not None:
            self.model = self._get_model(data, data_t)

    def _init(self, spec):
        self.model_name = spec.get('model_name') or spec.get('model')
        self.hyperparams = spec.get('hyperparams', dict())
        self.output_path = spec.get('output_path')
        self.inference_method = '?'
        self.iterations = spec.get('iterations', 1)

        self.write = spec.get('write', False)
        # **kwargs
        self.cfg = spec

    def _format_dataset(self, data, data_t):

        testset_ratio = self.cfg.get('testset_ratio')

        if 'text' in str(type(data)).lower():
            #if issubclass(type(data), DataBase):
            lgg.warning('check WHY and WHEN overflow in stirling matrix !?')
            print('debug why error and i get walue superior to 6000 in the striling matrix ????')
            if testset_ratio is None:
                data = data.data
                data_t = None
            else:
                data, data_t = data.cross_set(ratio=testset_ratio)
        elif 'network' in str(type(data)).lower():
            data_t = None
            if testset_ratio is None:
                data = data.data
            else:
                data = data.get_masked(testset_ratio)
        else:
            raise NotImplementedError('Data not understood')


        return data, data_t


    def _get_model(self, data=None, data_t=None):

        # Not all model takes data (Automata ?)
        data, data_t = self._format_dataset(data, data_t)

        kwargs = dict(data=data, data_t=data_t)

        if self.model_name in ('ilda', 'lda_cgs', 'immsb', 'mmsb_cgs'):
            model = self.loadgibbs_1(**kwargs)
        elif self.model_name in ('lda_vb'):
            self.model_name = 'ldafullbaye'
            model = self.lda_gensim(**kwargs)
        elif self.model_name in ('ilfm', 'ibp', 'ibp_cgs'):
            model = self.loadgibbs_2(**kwargs)
            model.normalization_fun = lambda x : 1/(1 + np.exp(-x))
        else:
            raise NotImplementedError()

        model.update_hyper(self.hyperparams)
        return model

    def fit(self, data=None):
        ''' if data is not None, create a new model instance.
            This is a batch mode. Future will be a online update

            Parameters
            ----------
            data : dataBase
        '''

        if data is not None:
            self.model = self._get_model(data)

        if hasattr(self.model, 'fit'):
            self.model.fit()

        if self.write:
            self.save()

        return


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
        #
        #

        ### Prediction Measures
        data = frontend.data

        # if modelNetwork ...
        res = self.model.predictMask(data)
        #elif modelText

        ### Data Measure
        if frontend is not None:
            data_prop = frontend.get_data_prop()
            res.update(data_prop)

        if self.write:
            frontend.save_json(res)
        else:
            lgg.debug(res)


    # Base class for Gibbs, VB ... ?
    def loadgibbs_1(self, **kwargs):
        delta = self.hyperparams.get('delta',1)
        alpha = self.hyperparams.get('alpha',1)
        gmma = self.hyperparams.get('gmma',1)

        hyper = self.cfg['hyper']
        assortativity = self.cfg.get('homo')
        hyper_prior = self.cfg.get('hyper_prior') # HDP hyper optimization
        K = self.cfg['K']

        model_name = self.model_name
        likelihood = kwargs.get('likelihood')
        data = kwargs['data']
        data_t = kwargs.get('data_t')

        if 'mmsb' in model_name:
            kernel = mmsb
        elif 'lda' in model_name:
            kernel = lda

        if likelihood is None:
            likelihood = kernel.Likelihood(delta,
                                           data,
                                           assortativity=assortativity)

        if model_name.split('_')[-1] == 'cgs':
            # Parametric case
            jointsampler = kernel.CGS(kernel.ZSamplerParametric(alpha,
                                                                likelihood,
                                                                K,
                                                                data_t=data_t))
        else:
            # Nonparametric case
            zsampler = kernel.ZSampler(alpha,
                                       likelihood,
                                       K_init=K,
                                       data_t=data_t)
            msampler = kernel.MSampler(zsampler)
            betasampler = kernel.BetaSampler(gmma,
                                             msampler)
            jointsampler = kernel.NP_CGS(zsampler,
                                         msampler,
                                         betasampler,
                                         hyper=hyper, hyper_prior=hyper_prior)

        return kernel.GibbsRun(jointsampler,
                               iterations=self.iterations,
                        output_path=self.output_path,
                               write=self.write,
                               data_t=data_t)

    def loadgibbs_2(self, **kwargs):
        sigma_w = 1.
        sigma_w_hyper_parameter = None #(1., 1.)
        alpha = self.hyperparams.get('alpha',1)

        alpha_hyper_parameter = self.cfg['hyper']
        assortativity = self.cfg.get('homo')
        K = self.cfg['K']

        model_name = self.model_name
        data = kwargs['data']
        data_t = kwargs.get('data_t')

        if '_cgs' in model_name:
            metropolis_hastings_k_new = False
        else:
            metropolis_hastings_k_new = True
            if self.cfg['homo'] == 2:
                raise NotImplementedError('Warning !: Metropolis Hasting not implemented with matrix normal. Exiting....')

        model = IBPGibbsSampling(assortativity,
                                 alpha_hyper_parameter,
                                 sigma_w_hyper_parameter,
                                 metropolis_hastings_k_new,
                                 iterations=self.iterations,
                                 output_path=self.output_path,
                                 write=self.write)
        model._initialize(data, alpha, sigma_w, KK=K)
        lgg.warn('Warning: K is IBP initialized...')
        #self.model._initialize(data, alpha, sigma_w, KK=None)
        return model

    # **kwargs !?
    def lda_gensim(self, data=None, data_t=None, id2word=None, save=False, model='ldamodel', load=False, updatetype='batch'):
        fname = self.output_path if self.write else None
        delta = self.hyperparams['delta']
        alpha = 'asymmetric'
        K = self.cfg['K']

        data = kwargs['data']
        data_t = kwargs.get('data_t')

        if load:
            return Models[model].LdaModel.load(fname)

        if hasattr(data, 'tocsc'):
            # is csr sparse matrix
            data = data.tocsc()
            data = gsm.matutils.Sparse2Corpus(data, documents_columns=False)
            if heldout_data is not None:
                heldout_data = heldout_data.tocsc()
                heldout_data = gsm.matutils.Sparse2Corpus(heldout_data, documents_columns=False)
        elif isanparray:
            # up tocsc ??!!! no !
            dense2corpus
        # Passes is the iterations for batch onlines and
        # iteration the max it in the gamma treshold test
        # loop Batch setting !
        if updatetype == 'batch':
            lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                         iterations=100, eval_every=None, update_every=None, passes=self.iterations, chunksize=200000,
                                         fname=fname, heldout_corpus=heldout_data)
        elif updatetype == 'online':
            lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                         iterations=100, eval_every=None, update_every=1, passes=1, chunksize=2000,
                                         fname=fname, heldout_corpus=heldout_data)

        if save:
            lda.expElogbeta = None
            lda.sstats = None
            lda.save(fname)
        return lda

    def initialization_test(self):
        ''' Measure perplexity on different initialization '''
        niter = 2
        pp = []
        likelihood = self.model.s.zsampler.likelihood
        for i in range(niter):
            self.model.s.zsampler.estimate_latent_variables()
            pp.append( self.model.s.zsampler.perplexity() )
            self.model = self.loadgibbs(self.model_name, likelihood)

        np.savetxt('t.out', np.log(pp))

    # Pickle class
    def save(self):
        fn = self.output_path + '.pk'
        ### Debug for non serializable variables
        #for u, v in vars(self.model).items():
        #    with open(f, 'w') as _f:
        #        try:
        #            pickle.dump(v, _f, protocol=pickle.HIGHEST_PROTOCOL )
        #        except:
        #            print 'not serializable here: %s, %s' % (u, v)
        self.model._f = None
        self.model.purge()
        # |
        # |
        # HOW TO called thid method recursvely from..
        to_remove = []
        for k, v in self.model.__dict__.items():
            if hasattr(v, 'func_name') and v.func_name == '<lambda>':
                to_remove.append(k)
            if str(v).find('<lambda>') >= 0:
                # python3 hook, nothing better ?
                to_remove.append(k)
            #elif type(k) is defaultdict:
            #    setattr(self.model, k, dict(v))

        for k in to_remove:
            try:
                delattr(self.model, k)
            except:
                pass

        lgg.info('Saving Model : %s' % fn)
        with open(fn, 'wb') as _f:
            return pickle.dump(self.model, _f, protocol=pickle.HIGHEST_PROTOCOL)

    # Debug classmethod and ecrasement d'object en jeux.
    #@classmethod
    def load(self, spec=None, init=False):
        if spec:
            self._init(spec)

        if init is True:
            model = self._get_model()
        else:
            if spec == None:
                fn = self.output_path + '.pk'
            else:
                fn = make_output_path(spec, 'pk')
            if not os.path.isfile(fn) or os.stat(fn).st_size == 0:
                print('No file for this model: %s' %fn)
                print('The following are available:')
                for f in model_walker(os.path.dirname(fn), fmt='list'):
                    print(f)
                return None
            lgg.info('Loading Model: %s' % fn)
            with open(fn, 'rb') as _f:
                try:
                    model =  pickle.load(_f)
                except:
                    # python 2to3 bug
                    _f.seek(0)
                    model =  pickle.load(_f, encoding='latin1')
        self.model = model
        return model


class ExpeManager(object):
    """ Mange set of experiments
        * Main loop consist of Corpuses and classe
        * a Tensor ?
    """
    def __init(self):
        pass


