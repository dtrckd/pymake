import os
import json, copy
import logging
from itertools import chain
from string import Template
from collections import defaultdict

import numpy as np

from pymake import GramExp


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

        self._force_load_data = expe.get('_force_load_data', True)
        self._force_save_data = expe.get('_force_save_data', True)

        self.corpus_name = expe.get('corpus')


    @classmethod
    def from_expe(cls):
        raise NotImplementedError

    @classmethod
    def _get_data(cls):
        ''' Raw data parsing/extraction. '''
        raise NotImplementedError

    @classmethod
    def _resolve_filename(cls, expe):
        input_path = expe._input_path

        if not os.path.exists(input_path):
            self.log.error("Corpus `%s' Not found." % (input_path))
            print('please run "fetch_networks"')
            self.data = None
            return

        if expe.corpus.endswith('.pk'):
            basename = expe.corpus
        else:
            basename = expe.corpus + '.pk'

        fn = os.path.join(input_path, basename)
        return fn

    @classmethod
    def _load_data(cls, *args, **kwargs):
        ''' Load preprocessed data. '''
        from pymake.io import load
        return load(*args, **kwargs)

    @classmethod
    def _save_data(cls, *args, **kwargs):
        ''' Save preprocessed data. '''
        from pymake.io import save
        return save(*args, **kwargs)

    #
    # Experimental Api
    #

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


    # frontendNetwork_nx
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



