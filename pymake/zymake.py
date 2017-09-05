#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


# __remove__ -> INTERNAL !
#from joblib import Parallel, delayed
import sys, multiprocessing

from pymake import GramExp


''' A Command line controler of Pymake '''



def main():

    zymake = GramExp.zymake()
    zyvar = zymake._conf

    if zyvar.get('simulate'):
        # same as show !
        zymake.simulate()

    ### Makes OUT Files
    lines = None
    line_prefix = ''
    if zyvar['_do'] == 'cmd':
        lines = zymake.make_commandline()
    elif zyvar['_do'] == 'path':
        lines = zymake.make_path(ftype=zyvar.get('_ftype', 'inf'), status=zyvar.get('_status'))
    elif zyvar['_do'] == 'show':
        zymake.simulate()
    elif zyvar['_do'] ==  'run':
        line_prefix = './zymake.py run'
        lines = zymake.execute()
    elif zyvar['_do'] == 'notebook':
        lines = zymake.notebook()
    elif zyvar['_do'] == 'update':
        zymake.update_index()
    elif zyvar['_do'] == 'init':
        zymake.init_folders()
    elif zyvar['_do'] == 'burn':
        # @todo; parallelize Pymake()
        raise NotImplementedError('What parallel strategy ?')
    else:

        if not 'do_list' in zyvar and zyvar['_do']:
            raise ValueError('Unknown command : %s' % zyvar['_do'])

        if 'model' == zyvar.get('do_list'):
            print (zymake.atomtable())
        elif 'model_topos' == zyvar.get('do_list'):
            print (zymake.atomtable(_type='topos'))
        elif 'script' == zyvar.get('do_list'):
            print(zymake.scripttable())
        elif 'expe' ==  zyvar.get('do_list'):
            print (zymake.spectable())
        else:
            print(zymake.help_short())
            if zyvar.get('do_list'):
                print ('Unknow options %s ?' % zyvar.get('do_list'))
        exit()

    if lines is None:
        # catch signal ?
        exit()

    if 'script' in zyvar:
        lines = [' '.join((line_prefix, l)).strip() for l in lines]

    zymake.simulate(halt=False, file=sys.stderr)
    print('\n'.join(lines), file=sys.stdout)

if __name__ == '__main__':
    main()

