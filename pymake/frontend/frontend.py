import os
import logging

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

    @staticmethod
    def get_input_path(expe):
        if '_input_path' in expe:
            return expe['_input_path']
        else:
            input_path =  GramExp.make_input_path(expe)
            expe['_input_path'] = input_path
            return input_path


    @classmethod
    def _resolve_filename(cls, expe):
        input_path = expe._input_path

        if not os.path.exists(input_path):
            cls.log.error("Corpus `%s' Not found." % (input_path))
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
         '''
        if self.data is None:
            return

        if self.expe.get('exponentiate'):
            MAX = 300
            self.data.ep['weights'].a = 2**self.data.ep['weights'].a
            self.data.ep['weights'].a[self.data.ep['weights'].a > MAX] = int(MAX)
            self.data.ep['weights'].a[self.data.ep['weights'].a < 0] = int(MAX)

        N = self.expe.get('N')
        if N is not None and N != 'all':
            N = int(N)
            self.log.debug('sampling dataset to N=%d ...' % N)
            self.sample(N)

        testset_ratio = self.expe.get('testset_ratio')
        if testset_ratio is not None:
            self.log.debug('Building testset ...')
            self.make_testset()


        if self.expe.get('noise'):
            self.make_noise()

        return


