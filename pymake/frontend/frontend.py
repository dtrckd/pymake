import os
import logging

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

    #
    # I/O Methods
    #

    @classmethod
    def from_expe(cls):
        raise NotImplementedError

    @classmethod
    def _extract_data(cls):
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

    def configure(self):
        ''' Configure the frontend Data.
        Try the following steps:
            1. Sample the corpus (expe.N),
            2. Build a testset/validation set (expe.testset_ratio & mask),
            3. build a sampling strategy (expe.sampling)
         '''
        if self.data is None:
            return

        N = self.expe.get('N')
        if N is not None and N != 'all':
            self.sample(N)

        testset_ratio = self.expe.get('testset_ratio')
        if testset_ratio is not None:
            self.make_testset(testset_ratio)

        sampling_strategy = self.expe.get('sampling')
        if sampling_strategy is not None:
            self.make_sampling(sampling_strategy)


        return


