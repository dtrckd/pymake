import os
import json, copy
from itertools import chain
from string import Template
from collections import defaultdict

import numpy as np

from pymake import GramExp

import logging

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

    log = logging.getLogger('root')

    def __init__(self, expe, load=False):

        # Load a .pk file for data(default: True if present)
        # + it faset
        # - some data features are not stored in .pk
        self._load_data = expe.get('_load_data', True)

        # Save a .pk file of data
        self._save_data = expe.get('_save_data', False)

        self.corpus_name = expe.get('corpus')
        self.model_name = expe.get('model')

        # Specific / @issue Object ?
        # How to handld not-defined variable ?
        # What categorie for object ??
        self.homo = int(expe.get('homo', 0))
        self.hyper_optimiztn = expe.get('hyper')
        self.clusters = None
        self.features = None

        self.true_classes = None

        ########
        ### Update settings Operations
        ########
        self.expe = expe
        expe['_data_type'] = self.bdir
        self.make_io_path()
        # There is some dynamic settings
        # Yes, use gramexp to setup path !!!
        # K, others ?

        self._data_file_format = None
        data_format = self.expe.get('_data_format', 'b')
        if data_format == 'w':
            self._net_type = 'weighted'
            self._dtype = int
        elif data_format == 'b':
            self._net_type = 'binary'
            self._dtype = bool
        else:
            raise NotImplemented('Network format unknwown: %s' % data_format)

        if load is True:
            self.load_data(randomize=False)

        # Copy Contructor in Python ?
        #if data is not None:
        #    self.update_data(data)

        # @Obsolete
        self.data_t = None

    def update_data(self):
        raise NotImplemented

    def make_io_path(self):
        ''' Write Path (for models results) in global settings '''
        self.output_path = GramExp.make_output_path(self.expe)
        self.input_path = GramExp.make_input_path(self.expe)

        # path can be updated by self.
        self.expe['_output_path'] = self.output_path
        self.expe['_input_path'] = self.input_path

    def update_spec(self, **spec):
        v = None
        if len(spec) == 1:
            k, v = list(spec.items())[0]
            setattr(self, k, v)
        self.expe.update(spec)
        return v

    @staticmethod
    def corpus_walker(path):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()
    def _get_corpus(self):
        raise NotImplementedError()

    def get_data_prop(self):
        prop = defaultdict()
        prop.update( {'corpus': self.corpus_name,
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
        #
        #
    @staticmethod
    def symmetrize(self, data=None):
        if data is None:
            return None
        data = np.triu(data) + np.triu(data, 1).T
        return

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

    @staticmethod
    def _save(data, fn, silent=False):
        GramExp.save(data, fn, silent)

    @staticmethod
    def _load(fn):
        return GramExp.load(fn)

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

        self.log.info('Saving json : %s' % fn)
        return json.dump(new_res, open(fn,'w'))
    def get_json(self):
        fn = self.output_path + '.json'
        self.log.info('Loading json frData ; %s' % fn)
        d = json.load(open(fn,'r'))
        return d
    def update_json(self, d):
        fn = self.output_path + '.json'
        res = json.load(open(fn,'r'))
        res.update(d)
        self.log.info('Updating json frData: %s' % fn)
        json.dump(res, open(fn,'w'))
        return fn


