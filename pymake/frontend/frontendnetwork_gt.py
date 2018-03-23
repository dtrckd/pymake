from .frontend import DataBase
from .drivers import DatasetDriver

import graph_tool as gt

from pymake.util.math import *



class frontendNetwork_gt(DataBase, DatasetDriver):

    def __init__(self, expe=None):
        super(frontendNetwork, self).__init__(expe)

        self._data_type = 'network'

    # @TODO: should be in Manager
    # and follows the load_model typo.
    def load_data(self, randomize=False):
        corpus_name = self.corpus_name

        # DB integration ?
        if corpus_name.startswith(('generator', 'graph')):
            format = 'graph'
        elif corpus_name in ('bench1'):
            raise NotImplementedError()
        elif corpus_name.startswith('facebook'):
            format = 'edges'
        elif corpus_name in ('manufacturing',):
            format = 'csv'
        elif corpus_name in ('fb_uc', 'emaileu'):
            format = 'txt'
        elif corpus_name in ('blogs','propro', 'euroroad'):
            format = 'dat'
        else:
            raise ValueError('Which corpus to Load; %s ?' % corpus_name)

        data = self.networkloader(corpus_name, format)
