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

    # @need an expe
    def completed(self, ext=None):
        ''' Show completed expe. '''
        if self.is_first_expe():
            self.D.n_exp_total = self.expe_size
            self.D.n_exp_completed = 0

        is_fitted = self.gramexp.make_output_path(self.expe, ext=ext, status='f')
        if is_fitted:
            self.D.n_exp_completed += 1
            self.log.debug(self.expe['_output_path'])


        if self.is_last_expe():
            table = OrderedDict([('completed', [self.D.n_exp_completed]),
                                 ('total', [self.D.n_exp_total]),
                                ])
            print (self.tabulate(table, headers='keys', tablefmt='simple', floatfmt='.3f'))


    # @need an expe
    def missing(self, ext=None):
        ''' Show missing expe. '''
        if self.is_first_expe():
            self.D.n_exp_total = self.expe_size
            self.D.n_exp_missing = 0

        is_fitted = self.gramexp.make_output_path(self.expe, ext=ext, status='f')
        if not is_fitted:
            self.D.n_exp_missing += 1
            self.log.debug(self.expe['_output_path'])


        if self.is_last_expe():
            table = OrderedDict([('missing', [self.D.n_exp_missing]),
                                 ('total', [self.D.n_exp_total]),
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

    def move(self, *args):
        import glob
        import shutil
        assert(len(args)>0)

        expe = self.expe
        new_expe = expe.copy()

        for o in args:
            k,v = o.split('=')
            new_expe[k] = v

        opath = self.output_path
        npath = self.gramexp.make_output_path(new_expe)

        if self.is_first_expe():
            self.log.info('Moving files for request: %s' % args)
            self.D.mesg = []

        for ofn in glob.glob(opath+'.*'):
            ext = ofn[len(opath):]
            nfn = npath + ext
            self.D.mesg.append("`%s' -> `%s'" % (ofn, nfn))
            shutil.move(ofn, nfn)

        if self.is_last_expe():
            self.log.debug('\n'.join(self.D.mesg))
            self.log.info('%d files moved.' % len(self.D.mesg))



