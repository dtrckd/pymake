# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle, json, copy
from itertools import chain
from string import Template
from collections import defaultdict
import logging
lgg = logging.getLogger('root')

import numpy as np

from .frontend_io import *


''' Actually this is more the Backend ...! '''


class Object(object):
    """ Implement a mathematical object manipulation philosophy,
        WIth a high level view of object set as topoi.
        * return None for errorAtributes
        * memorize all input as attribute by default
    """
    # @todo: catch attributeError on config, and print possibilities if possible
    # (ie the method assciated to the key in the object in getattr ? --> tab completion)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    #def __getattr__(self, attr):
    #    if type(attr) is not str:
    #        lgg.error('Error attribute type: %s' % (attr))
    #        return None

    #    if hasattr(self,attr):
    #        return getattr(self, attr)
    #    else:
    #        if hasattr(self, 'get_'+attr):
    #            lgg.warning('get %s from class method get_' % (str(attr)))
    #            f = self.getattr(self, 'get_'+attr)
    #            return f()
    #        else:
    #            # find the name of the chil class on this catch
    #            lgg.warning('attributes `%s\' is Non-Existent' % (str(attr)))
    #            return None

class DataBase(object):
    """ Root Class for Frontend Manipulation over Corpuses and Models.

        Given Data Y, and Model M = {\theta, \Phi}
        E[Y] = \theta \phi^T

        Fonctionality are of the frontend decline as:
        1. Frontend for model/algorithm I/O,
        2. Frontend for Corpus Information, and Result Gathering for
            Machine Learning Models.
        3. Data Analisis and Prediction..

        load_corpus -> load_text_corpus -> text_loader
        (frontent)  ->   (choice)       -> (adapt preprocessing)

    """
    ### @Debug :
    #   * update config path !
    #   * Separate attribute of the frontend: dataset / Learning / IHM ...

    # review that:
    #    * separate better load / save and preprocessing (input can be file or array...)
    #    * view of file confif.... and path creation....

    def __init__(self, config, data=None):
        if config.get('seed'):
            #np.random.seed(config.get('seed'))
            np.random.set_state(self.load('.seed'))
        self.seed = np.random.get_state()
        self.save(self.seed, '.seed')
        self.cfg = config
        config['data_type'] = self.bdir
        self._load_data = config.get('load_data')
        self._save_data = config.get('save_data')

        self.corpus_name = config.get('corpus_name') or config.get('corpus')
        self.model_name = config.get('model_name')

        # Specific / @issue Object ?
        # How to handld not-defined variable ?
        # What categorie for object ??
        self.homo = int(config.get('homo', 0))
        self.hyper_optimiztn = config.get('hyper')
        self.clusters = None
        self.features = None

        self.true_classes = None
        self.data = data
        self.data_t = None

        # Read Directory
        #self.make_output_path()

        # self._init()
        # self.data = self.load_data(spec)
        if data is not None:
            self.update_data(data)

    def update_data(self):
        raise NotImplemented

    def make_output_path(self):
        # Write Path (for models results)
        self.basedir, self.output_path = make_output_path(self.cfg)
        self.cfg['output_path'] = self.output_path

    def update_spec(self, **spec):
        v = None
        if len(spec) == 1:
            k, v = list(spec.items())[0]
            setattr(self, k, v)
        self.cfg.update(spec)
        return v

    @staticmethod
    def corpus_walker(path):
        raise NotImplementedError()

    #######
    # How to get a chlidren class from root class !?
    # See also: load_model() return super('child') ...
    #######
    def load_data(self):
        raise NotImplementedError()

    # convert ndarray to list.
    def save_json(self, res):
        """ Save a dictionnary in json"""
        fn = self.output_path + '.json'
        new_res = copy.copy(res)
        for k, v  in new_res.items():
            # Go at two level deeper, no more !
            if type(v) is dict:
                for kk, vv  in v.items():
                    if hasattr(vv, 'tolist'):
                        new_res[k][kk] = vv.tolist()
            if hasattr(v, 'tolist'):
                new_res[k] = v.tolist()
        return json.dump(new_res, open(fn,'w'))
    def get_json(self):
        fn = self.output_path + '.json'
        d = json.load(open(fn,'r'))
        return d
    def update_json(self, d):
        fn = self.output_path + '.json'
        res = json.load(open(fn,'r'))
        res.update(d)
        lgg.info('updating json: %s' % fn)
        json.dump(res, open(fn,'w'))
        return fn

    def get_data_prop(self):
        prop = defaultdict()
        prop.update( {'corpus_name': self.corpus_name,
                'instances' : self.data.shape[1] })
        return prop

    # Template for corpus information: Instance, Nnz, features etx
    def template(self, dct, templ):
        return Template(templ).substitute(dct)

    def shuffle_instances(self):
        index = np.arange(np.shape(self.data)[0])
        np.random.shuffle(index)
        self.data =  self.data[index, :]
        #if hasattr(self.data, 'A'):
        #    data = self.data.A
        #    np.random.shuffle(data)
        #    self.data = sp.sparse.csr_matrix(data)
        #else:
        #    np.random.shuffle(self.data)

    def shuffle_features(self):
        raise NotImplemented

    # Return a vector with document generated from a count matrix.
    # Assume sparse matrix
    @staticmethod
    def sparse2stream(data):
        #new_data = []
        #for d in data:
        #    new_data.append(d[d.nonzero()].A1)
        bow = []
        for doc in data:
            # Also, see collections.Counter.elements() ...
            bow.append( np.array(list(chain(* [ doc[0,i]*[i] for i in doc.nonzero()[1] ]))))
        bow = np.array(bow)
        #map(np.random.shuffle, bow)
        return bow

    # Pickle class
    @staticmethod
    def save(data, fn):
        fn = fn + '.pk'
        with open(fn, 'wb') as _f:
            return pickle.dump(data, _f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fn):
        fn = fn + '.pk'
        lgg.debug('opening file: %s' % fn)
        with open(fn, 'r') as _f:
            return pickle.load(_f)

    @staticmethod
    def symmetrize(self, data=None):
        if data is None:
            return None
        data = np.triu(data) + np.triu(data, 1).T


class ModelBase(object):
    """"  Root Class for all the Models.

    * Suited for unserpervised model
    * Virtual methods for the desired propertie of models
    """
    default_settings = {
        'snapshot_interval' : 100, # UNUSED
        'write' : False,
        'output_path' : 'tm-output',
        'csv_typo' : None,
        'fmt' : None,
        'iterations' : 1
    }
    def __init__(self, **kwargs):
        """ Model Initialization strategy:
            1. self lookup from child initalization
            2. kwargs lookup
            3. default value
        """
        self.samples = [] # actual sample
        self._samples    = [] # slice to save

        for k, v in self.default_settings.items():
            self._init(k, kwargs, v)

        if self.output_path and self.write:
            import os
            bdir = os.path.dirname(self.output_path)
            fn = os.path.basename(self.output_path)
            try: os.makedirs(bdir)
            except: pass
            self.fname_i = bdir + '/inference-' + fn.split('.')[0]
            self._f = open(self.fname_i, 'wb')
            self._f.write((self.csv_typo + '\n').encode('utf8'))

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

    def write_some(self, samples, buff=20):
        """ Write data with buffer manager """
        f = self._f
        fmt = self.fmt

        if samples is None:
            buff=1
        else:
            self._samples.append(samples)

        if len(self._samples) >= buff:
            #samples = np.array(self._samples)
            samples = self._samples
            np.savetxt(f, samples, fmt=str(fmt))
            f.flush()
            self._samples = []

    # try on output_path i/o error manage fname_i
    def load_some(self, iter_max=None):
        filen = self.fname_i
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        if iter_max:
            data = data[:iter_max]
        # Ignore Comments
        data = [re.sub("\s\s+" , " ", x.strip()) for l,x in enumerate(data) if not x.startswith(('#', '%'))]

        #ll_y = [row.split(sep)[column] for row in data]
        #ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))
        return data

    def close(self):
        if not hasattr(self, '_f'):
            return
        # Write remaining data
        if self._samples:
            self.write_some(None)
        self._f.close()

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
            return self.reduce_latent()

    def purge(self):
        """ Remove variable that are non serializable. """
        return

    def update_hyper(self):
        lgg.error('no method to update hyperparams')
        return

    def get_hyper(self):
        lgg.error('no method to get hyperparams')
        return
    # Just for MCMC ?():
    def reduce_latent(self):
        raise NotImplementedError
    def communities_analysis():
        raise NotImplementedError
    def generate(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def fit(self):
        raise NotImplementedError
    def link_expectation(self):
        raise NotImplementedError
    def get_clusters(self):
        raise NotImplementedError






