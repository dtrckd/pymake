from pymake.frontend.manager import ModelManager, FrontendManager
from pymake import ExpeFormat

from collections import OrderedDict

import matplotlib.pyplot as plt
from pymake.util.utils import colored



USAGE = """\
----------------
Manage the data : This script is part of the repo/ml of pymake.
----------------
"""



class Data(ExpeFormat):

    _default_expe = { '_expe_silent' : True }

    # @need an expe
    def missing(self, _type='pk'):
        ''' Show missing expe. '''
        if self.is_first_expe():
            self.gramexp.n_exp_total = self.expe_size
            self.gramexp.n_exp_missing = 0

        is_fitted = self.gramexp.make_output_path(self.expe, _type=_type, status='f')
        if not is_fitted:
            self.gramexp.n_exp_missing += 1
            self.log.debug(self.expe['_output_path'])


        if self.is_last_expe():
            table = OrderedDict([('missing', [self.gramexp.n_exp_missing]),
                                 ('total', [self.gramexp.n_exp_total]),
                                ])
            print (self.tabulate(table, headers='keys', tablefmt='simple', floatfmt='.3f'))


    # @need an expe
    def completed(self, _type='pk'):
        ''' Show completed expe. '''
        if self.is_first_expe():
            self.gramexp.n_exp_total = self.expe_size
            self.gramexp.n_exp_completed = 0

        is_fitted = self.gramexp.make_output_path(self.expe, _type=_type, status='f')
        if is_fitted:
            self.gramexp.n_exp_completed += 1
            self.log.debug(self.expe['_output_path'])


        if self.is_last_expe():
            table = OrderedDict([('completed', [self.gramexp.n_exp_completed]),
                                 ('total', [self.gramexp.n_exp_total]),
                                ])
            print (self.tabulate(table, headers='keys', tablefmt='simple', floatfmt='.3f'))



    def topo(self):
        print('''' Todo Topo:

                1. get all _typo in spec,
                2. parse all file (pk ou inf ?)
                3. classify all expe accordinf to : refdir, _name, corpus, model
                4. tabulate.
             ''')
        pass



