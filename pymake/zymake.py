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

    if zyvar.get('simulate') and not zyvar['_do'] in ['run', 'runpara']:
        # same as show !
        zymake.simulate()

    ### Makes OUT Files
    lines = None
    line_prefix = ''
    if zyvar['_do'] == 'init':
        zymake.init_folders()
    elif zyvar['_do'] == 'update':
        zymake.update_index()
    elif zyvar['_do'] == 'show':
        zymake.simulate()
    elif zyvar['_do'] ==  'run':
        lines = zymake.execute()
        zymake.pushcmd2hist()
    elif zyvar['_do'] == 'runpara':
        lines = zymake.execute_parallel()
        zymake.pushcmd2hist()
    elif zyvar['_do'] == 'cmd':
        lines = zymake.make_commandline()
    elif zyvar['_do'] == 'path':
        lines = zymake.make_path(ftype=zyvar.get('_ftype', 'inf'), status=zyvar.get('_status'))
    elif zyvar['_do'] == 'hist':
        lines = zymake.show_history()
    elif zyvar['_do'] == 'notebook': # @Todo
        lines = zymake.notebook()
    else: # list things

        if not 'do_list' in zyvar and zyvar['_do']:
            raise ValueError('Unknown command : %s' % zyvar['_do'])

        if 'spec' == zyvar.get('do_list'):
            print (zymake.spectable())
        elif 'model' == zyvar.get('do_list'):
            print (zymake.modeltable())
        elif 'model_topos' == zyvar.get('do_list'):
            print (zymake.modeltable(_type='topos'))
        elif 'script' == zyvar.get('do_list'):
            print(zymake.scripttable())
        elif 'topo' ==  zyvar.get('do_list'):
            print (zymake.topotable())
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

