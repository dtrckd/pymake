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

    def __init__(self, expe):
        self.expe = expe

        # Load a .pk file of **preprocessed** data(default: True if present)
        self._load_data = expe.get('_load_data', True)

        # Save a .pk file of data
        self._save_data = expe.get('_save_data', False)

        self.corpus_name = expe.get('corpus')
        self.model_name = expe.get('model')

        # Specific / @issue Object ?
        # How to handle undefined variable ?
        # What category for object ??
        self.homo = int(expe.get('homo', 0))
        self.clusters = None
        self.features = None
        self.true_classes = None
        self._data_file_format = None

        # @Obsolete
        self.data_t = None

    def update_data(self):
        raise NotImplemented

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
        ''' inp-place symmetrization. '''
        if data is None:
            return None
        data = np.triu(data) + np.triu(data, 1).T

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
    def load(*args, **kwargs):
        from pymake.io import load
        return load(*args, **kwargs)

    @staticmethod
    def save(*args, **kwargs):
        from pymake.io import save
        return save(*args, **kwargs)


