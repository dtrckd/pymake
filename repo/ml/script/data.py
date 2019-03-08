import os
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


    # @need an expe
    def incomplete(self, ext='inf'):
        ''' Show not terminated expe. '''
        if self.is_first_expe():
            self.D.n_exp_total = self.expe_size
            self.D.n_exp_missing = 0

        is_fitted = self.gramexp.make_output_path(self.expe, ext=ext, status='f')
        _file = self.expe['_output_path'] +'.' + ext
        try:
            if not is_fitted:
                flag = False
            else:
                flag = list(filter(None,open(_file).read().split('\n')))[-1].split()[-1]

            is_incomplete = flag != 'terminated'
        except FileNotFoundError as e:
            is_incomplete = True

        if is_incomplete:
            self.D.n_exp_missing += 1
            self.log.debug(self.expe['_output_path'])


        if self.is_last_expe():
            table = OrderedDict([('incomplete', [self.D.n_exp_missing]),
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

    def move(self, *args, copy=False):
        ''' move a experiences files (all extension)
            (simulta by default. Use -f to force the operation)

            *args are sequence of parameters to change in the filename with the syntax:
                -x move foo=bar
            which means that the paramters {foo} in the output_path will be modified with the value {bar}
        '''
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
            self.D.num_expe = 0
            self.D.num_total = 0

        self.D.num_total += 1

        textm = 'copying' if copy  else 'moving'
        cwd = os.getenv('PWD')
        pwd_len = len(cwd)

        flag = False
        for ofn in glob.glob(opath+'.*'):
            ext = ofn[len(opath):]
            nfn = npath + ext
            self.D.mesg.append("%s `%s' -> `%s'" % (textm, './'+ofn[pwd_len+1:], './'+nfn[pwd_len+1:]))

            if expe.get('force'):
                if copy is True:
                    shutil.copyfile(ofn, nfn)
                else:
                    shutil.move(ofn, nfn)
                flag = True

        if flag:
            self.D.num_expe +=1

        if self.is_last_expe():
            if not expe.get('force'):
                print('*** Simulation (need force option.) ***')
            textm = 'copied' if copy else 'moved'
            self.log.debug('\n'.join(self.D.mesg)+'\n')
            self.log.info('%d/%d expe %s (%d files).' % (self.D.num_expe, self.D.num_total, textm, len(self.D.mesg)))


    def copy(self, *args):
        ''' like move but copy files instead.'''

        self.move(*args, copy=True)

