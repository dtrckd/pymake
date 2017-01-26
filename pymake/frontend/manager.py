# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import logging
lgg = logging.getLogger('root')

# Frontend Manager Utilities
from .frontendtext import frontendText
from .frontendnetwork import frontendNetwork
from .frontend_io import *

# Model Manager Utilities
import numpy as np
import pickle, json # presence of this module here + in .frontend not zen


# This is a mess, hormonize things with models
from models.hdp import mmsb, lda
from models.ibp.ilfm_gs import IBPGibbsSampling

#### @Debug/temp modules name changed in pickle model
from models import hdp, ibp
sys.modules['hdp'] = hdp
sys.modules['ibp'] = ibp
###

try:
    sys.path.insert(1, '../../gensim')
    import gensim as gsm
    from gensim.models import ldamodel, ldafullbaye
    Models = {'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}
except:
    pass

class FrontendManager(object):
    """ Utility Class who aims at mananing the frontend at the higher level.
    """
    @staticmethod
    def get(config):
        """ Return: The frontend suited for the given configuration"""
        corpus = config.get('corpus_name') or config.get('corpus')
        corpus_typo = {'network': ['facebook','generator', 'bench', 'clique', 'fb_uc', 'manufacturing'],
                       'text': ['reuter50', 'nips12', 'nips', 'enron', 'kos', 'nytimes', 'pubmed', '20ngroups', 'odp', 'wikipedia', 'lucene']}

        frontend = None
        for key, cps in corpus_typo.items():
            if corpus.startswith(tuple(cps)):
                if key == 'text':
                    frontend = frontendText(config)
                    break
                elif key == 'network':
                    frontend = frontendNetwork(config)
                    break

        if frontend is None:
            raise ValueError('Unknown Corpus `%s\'!' % corpus)
        return frontend


class ModelManager(object):
    """ Utility Class for Managing I/O and debugging Models
    """
    def __init__(self, data=None, config=None, data_t=None):
        if data is None:
            self.data = np.zeros((1,1))
        else:
            self.data = data
        self.data_t = data_t

        self._init(config)

        if self.config.get('load_model'):
            return
        if not self.model_name:
            return

        if data is not None:
            self.model = self.get_model(config)

    # Base class for Gibbs, VB ... ?
    def loadgibbs_1(self, target, likelihood=None):
        delta = self.hyperparams.get('delta',1)
        alpha = self.hyperparams.get('alpha',1)
        gmma = self.hyperparams.get('gmma',1)
        hyper = self.config['hyper']
        hyper_prior = self.config.get('hyper_prior') # HDP hyper optimization

        symmetric = self.config.get('symmetric',False)
        assortativity = self.config.get('homo')
        K = self.K

        if 'mmsb' in target:
            kernel = mmsb
        elif 'lda' in target:
            kernel = lda

        if likelihood is None:
            likelihood = kernel.Likelihood(delta,
                                           self.data,
                                           symmetric=symmetric,
                                           assortativity=assortativity)

        if target.split('_')[-1] == 'cgs':
            # Parametric case
            jointsampler = kernel.CGS(kernel.ZSamplerParametric(alpha,
                                                                likelihood,
                                                                K,
                                                                data_t=self.data_t))
        else:
            # Nonparametric case
            zsampler = kernel.ZSampler(alpha,
                                       likelihood,
                                       K_init=K,
                                       data_t=self.data_t)
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
                               data_t=self.data_t)

    def loadgibbs_2(self, model_name):
        alpha_hyper_parameter = self.config['hyper']
        symmetric = self.config.get('symmetric',False)
        assortativity = self.config.get('homo')
        K = self.K
        # Hyper parameter init
        alpha = self.hyperparams.get('alpha',1)
        sigma_w = 1.
        sigma_w_hyper_parameter = None #(1., 1.)

        if '_cgs' in model_name:
            metropolis_hastings_k_new = False
        else:
            metropolis_hastings_k_new = True
            if self.config['homo'] == 2:
                raise NotImplementedError('Warning !: Metropolis Hasting not implemented with matrix normal. Exiting....')

        model = IBPGibbsSampling(symmetric,
                                 assortativity,
                                 alpha_hyper_parameter,
                                 sigma_w_hyper_parameter,
                                 metropolis_hastings_k_new,
                                 iterations=self.iterations,
                                 output_path=self.output_path,
                                 write=self.write)
        model._initialize(self.data, alpha, sigma_w, KK=K)
        lgg.warn('Warning: K is IBP initialized...')
        #self.model._initialize(data, alpha, sigma_w, KK=None)
        return model

    def lda_gensim(self, id2word=None, save=False, model='ldamodel', load=False, updatetype='batch'):
        fname = self.output_path if self.write else None
        iter = self.config['iterations']
        data = self.data
        heldout_data = self.data_t
        delta = self.hyperparams['delta']
        #alpha = self.hyperparams['alpha']
        alpha = 'asymmetric'
        K = self.K
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
        # Passes is the iterations for batch onlines and iteration the max it in the gamma treshold test loop
        # Batch setting !
        if updatetype == 'batch':
            lda = Models[model].LdaModel(data, id2word=id2word, num_topics=K, alpha=alpha, eta=delta,
                                         iterations=100, eval_every=None, update_every=None, passes=iter, chunksize=200000,
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

    def fit(self):
        if hasattr(self.model, 'fit'):
            self.model.fit()

    def predict(self, frontend):
        if not hasattr(self.model, 'predict'):
            print('No predict method for self._name_ ?')
            return

        if self.data_t == None and not hasattr(self.data, 'mask') :
            print('No testing data for prediction ?')
            return

        ### Prediction Measures
        res = self.model.predict()

        ### Data Measure
        data_prop = frontend.get_data_prop()
        res.update(data_prop)

        if self.write:
            frontend.save_json(res)
            self.save()
        else:
            lgg.debug(res)

    # Measure perplexity on different initialization
    def init_loop_test(self):
        niter = 2
        pp = []
        likelihood = self.model.s.zsampler.likelihood
        for i in range(niter):
            self.model.s.zsampler.estimate_latent_variables()
            pp.append( self.model.s.zsampler.perplexity() )
            self.model = self.loadgibbs(self.model_name, likelihood)

        print(self.output_path)
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
            if str(v).find('<lambda>'):
                # python3 hook, nothing better ?
                to_remove.append(k)
            #elif type(k) is defaultdict:
            #    setattr(self.model, k, dict(v))

        for k in to_remove:
            try:
                delattr(self.model, k)
            except:
                pass

        with open(fn, 'wb') as _f:
            return pickle.dump(self.model, _f, protocol=pickle.HIGHEST_PROTOCOL)

    # Debug classmethod and ecrasement d'object en jeux.
    #@classmethod
    def load(self, spec=None, init=False):
        if spec:
            self._init(spec)

        if init is True:
            model = self.get_model(spec)
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
            lgg.debug('opening file: %s' % fn)
            with open(fn, 'rb') as _f:
                lgg.info('loading model: %s' % fn)
                try:
                    model =  pickle.load(_f)
                except:
                    # python 2to3 bug
                    _f.seek(0)
                    model =  pickle.load(_f, encoding='latin1')
        self.model = model
        return model

    def _init(self, spec):
        self.model_name = spec.get('model_name') or spec.get('model')
        #models = {'ilda' : HDP_LDA_CGS,
        #          'lda_cgs' : LDA_CGS, }
        self.hyperparams = spec.get('hyperparams', dict())
        self.output_path = spec.get('output_path')
        self.K = spec.get('K')
        self.inference_method = '?'
        self.iterations = spec.get('iterations', 0)

        self.write = spec.get('write', False)
        # **kwargs
        self.config = spec

    def get_model(self, spec):
        if self.model_name in ('ilda', 'lda_cgs', 'immsb', 'mmsb_cgs'):
            model = self.loadgibbs_1(self.model_name)
        elif self.model_name in ('lda_vb'):
            model = self.lda_gensim(model='ldafullbaye')
        elif self.model_name in ('ilfm', 'ibp', 'ibp_cgs'):
            model = self.loadgibbs_2(self.model_name)
            model.normalization_fun = lambda x : 1/(1 + np.exp(-x))
        else:
            raise NotImplementedError()

        model.update_hyper(self.hyperparams)
        return model


class ExpeManager(object):
    """ Mange set of experiments
        * Main loop consist of Corpuses and classe
        * a Tensor ?
    """
    def __init(self):
        pass


